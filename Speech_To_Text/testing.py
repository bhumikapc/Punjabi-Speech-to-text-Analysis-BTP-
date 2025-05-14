# Import necessary libraries
import os
import torch
import torchaudio
import pandas as pd
from jiwer import wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------- CONFIGURATION -----------

# Path to your fine-tuned Whisper model directory
model_path = './my_finetuned_model'  # --> to test fine-tuned model
# model_name = "DrishtiSharma/whisper-large-v2-punjabi-700-steps" #--> to test original

# Dataset root directory
dataset_path = './'

# Path to test file (TSV or TXT containing audio file names and transcripts)
test_txt_path = os.path.join(dataset_path, 'test.txt')

# Directory to search for audio files
search_root = os.path.join(dataset_path, 'concatenated_clips/')

# Directory to save predicted transcript text files
transcript_output_dir = os.path.join(dataset_path, 'predicted_transcripts_finetune_ldcil') #--> for saving fine-tuned predicted transcripts
#transcript_output_dir = os.path.join(dataset_path, 'predicted_transcripts_original_ldcil') #--> for saving original predicted transcripts
os.makedirs(transcript_output_dir, exist_ok=True)  # Create output dir if it doesn't exist

# Set device for inference (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- LOAD MODEL -----------

print("Loading model and processor...")
try:
    # Load the processor and model from the specified path (fine-tuned or pretrained)
    processor = WhisperProcessor.from_pretrained(model_path) #..> to test fine-tuned
    #processor = WhisperProcessor.from_pretrained(model_name)  #--> to test original
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device) #--> to test fine-tuned
    #model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device) #--> to test original
    model.eval()  # Set model to evaluation mode
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model/processor: {e}")
    exit()  # Stop execution if loading fails

# ----------- LOAD TEST DATA -----------

print(f"Loading test data from: {test_txt_path}")
try:
    # Read the test file (tab-separated: [audio_file] \t [transcript])
    test_df = pd.read_csv(test_txt_path, sep="\t", header=None, names=["audio_file", "transcript"])
    test_df = test_df.head(500)  # Optional: limit to first 500 samples for testing
    print(f"Loaded {len(test_df)} test samples.")
except FileNotFoundError:
    print(f"Test data file not found: {test_txt_path}")
    exit()
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# Optional memory optimization for CUDA
torch.cuda.empty_cache()
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------- FUNCTION: FIND AUDIO FILE -----------

def find_audio_file(filename, root_folder):
    """Search for a .wav audio file within root_folder and subdirectories."""
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file == filename or os.path.splitext(file)[0] == os.path.splitext(filename)[0]:
                if file.lower().endswith(".wav"):
                    return os.path.join(dirpath, file)
    return None  # Return None if file not found

# ----------- INFERENCE + WER CALCULATION -----------

individual_wers = []  # List to store WER per sample
processed_files_count = 0  # Count successfully processed files

print("\n🔍 Computing WER for each audio file and saving transcripts:\n")

# Loop through each test sample
for i, row in test_df.iterrows():
    file_key = row["audio_file"].strip()
    base_name_for_txt = os.path.splitext(os.path.basename(file_key))[0]  # For naming output file

    # Locate the audio file
    path = find_audio_file(file_key, search_root)
    ground_truth = str(row["transcript"])  # Ground truth transcript as string

    if path is None or not os.path.exists(path):
        print(f"[{i+1:03d}/{len(test_df):03d}] ❌ File not found for key: {file_key} (Searched in: {search_root})")
        continue  # Skip if file not found

    try:
        # Load and resample audio to 16 kHz
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # Convert stereo to mono if needed
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert waveform to 1D numpy array and preprocess with WhisperProcessor
        input_audio_np = waveform.squeeze().numpy()
        inputs = processor(input_audio_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Set forced decoder language prompts (e.g., for Punjabi)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="punjabi", task="transcribe")

        # Run the model and generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            decoded_predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            predicted_text = decoded_predictions[0] if decoded_predictions else ""
            predicted_text = str(predicted_text)  # Ensure string format

        # Calculate Word Error Rate (WER)
        error = wer(ground_truth, predicted_text)
        individual_wers.append(error)
        processed_files_count += 1

        # Print result for this sample
        print(f"[{i+1:03d}/{len(test_df):03d}] File: {os.path.basename(path)}")
        print(f"    Ground truth : {ground_truth}")
        print(f"    Prediction   : {predicted_text}")
        print(f"    WER          : {error:.3f}")

        # Save predicted transcript to text file
        output_txt_filename = f"{base_name_for_txt}.txt"
        output_txt_path = os.path.join(transcript_output_dir, output_txt_filename)
        try:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(predicted_text)
            print(f"    💾 Saved transcript to: {output_txt_path}\n")
        except Exception as e_save:
            print(f"    ⚠️ Error saving transcript to {output_txt_path}: {e_save}\n")

    except Exception as e_process:
        print(f"[{i+1:03d}/{len(test_df):03d}] ❌ Error processing file {file_key} (Path: {path}): {e_process}\n")

# ----------- SUMMARY -----------

if processed_files_count > 0:
    average_wer = sum(individual_wers) / len(individual_wers)
    print(f"\n📊 Average WER on {processed_files_count} processed samples: {average_wer:.3f}")
    print(f"📂 All successfully predicted transcripts have been saved to: {transcript_output_dir}")
else:
    print("\n📊 No audio files were successfully processed. No WER calculated.")
    if len(test_df) > 0:
         print(f"   Tried to process {len(test_df)} files from the input list.")
    print(f"   Transcript output directory (may be empty): {transcript_output_dir}")
