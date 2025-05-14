import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Trainer, TrainingArguments
import torchaudio
from accelerate import Accelerator
import torch.nn as nn


# Set up paths
dataset_path = './'
clips_path = os.path.join(dataset_path, 'concatenated_clips/')
transcript_file = os.path.join(dataset_path, 'train.txt')

# Load transcripts
transcripts = pd.read_csv(transcript_file, sep='\t', header=None, names=['audio_file', 'transcript'])
transcripts['audio_file'] = transcripts['audio_file'].apply(
    lambda x: os.path.join(clips_path, x.strip() + '.wav') if not x.endswith('.wav') else os.path.join(clips_path, x.strip())
)
#transcripts = transcripts.head(1000)
data = Dataset.from_pandas(transcripts)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Preprocessing: Load audio
def preprocess_data(example):
    audio_path = example['audio_file']
    if not os.path.exists(audio_path):
        return None
    try:
        audio_input, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if needed
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)

        # Resample if not at 16000 Hz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_input = resampler(audio_input)

        audio_input = audio_input.squeeze().numpy()
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    return {'audio': audio_input, 'text': example['transcript']}

data = data.map(preprocess_data, remove_columns=["audio_file", "transcript"])
data = data.filter(lambda example: example is not None)

#Load model and processor
model_name = "DrishtiSharma/whisper-large-v2-punjabi-700-steps"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final output projection layer
model.proj_out.weight.requires_grad = True

# Confirm trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Optional wrapper (not strictly needed if no decoder modifications)
class WhisperModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features=None, labels=None):
        return self.model(input_features=input_features, labels=labels)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

wrapped_model = WhisperModelWrapper(model)

# Preprocessing function
def preprocess_function(examples):
    inputs = processor.feature_extractor(
        examples["audio"], sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=3000
    )
    input_features = inputs.input_features
    padding_needed = 3000 - input_features.shape[-1]
    if padding_needed > 0:
        input_features = torch.nn.functional.pad(input_features, (0, padding_needed))
    labels = processor.tokenizer(
        examples["text"], return_tensors="pt", padding=True, truncation=True
    ).input_ids
    return {
        "input_features": input_features.squeeze(0).to(torch.float32),
        "labels": labels.squeeze(0),
    }

train_dataset = data.map(preprocess_function)

# Collator
def collate_fn(batch):
    input_features = [torch.tensor(x["input_features"]) for x in batch]
    input_features = torch.stack(input_features)

    labels = [torch.tensor(x["labels"]) for x in batch]
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    return {"input_features": input_features, "labels": labels}

# Training args
training_args = TrainingArguments(
    output_dir="./results1",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    gradient_accumulation_steps=64,
    gradient_checkpointing=True,
    fp16=True,
    optim="adamw_torch",
    logging_steps=1,
save_strategy="epoch",
report_to="none",
    ddp_find_unused_parameters=False,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = get_last_checkpoint("./results")
print("Resuming from:", last_checkpoint)

trainer.train(resume_from_checkpoint=last_checkpoint)

#trainer.train()
trainer.save_model("./my_finetuned_model1")
processor.save_pretrained("./my_finetuned_model1")



