import os
import torch
import torchaudio
import pandas as pd
from jiwer import wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------- CONFIG -----------
# path to your finetuned Whisper model
model_path = './my_finetuned_model' #--> to test fine-tuned
# model_name = "DrishtiSharma/whisper-large-v2-punjabi-700-steps" #--> to test original
dataset_path = './'
test_txt_path = os.path.join(dataset_path, 'test.txt')  # path to test.tsv file
search_root = os.path.join(dataset_path, 'concatenated_clips/')   # root directory to search audio files in

# New: Directory to save predicted transcripts
transcript_output_dir = os.path.join(dataset_path, 'predicted_transcripts_finetune_ldcil') #--> for saving fine tuned predicted transcripts
#transcript_output_dir = os.path.join(dataset_path, 'predicted_transcripts_original_ldcil') #--> for saving original predicted transcripts
os.makedirs(transcript_output_dir, exist_ok=True) # Ensure the directory exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- LOAD MODEL -----------

print("Loading model and processor...")
try:
    processor = WhisperProcessor.from_pretrained(model_path) #--> to test fine tuned
    #processor = WhisperProcessor.from_pretrained(model_name)  #--> to test original
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device) #--> to test fine tuned
    #model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device) #--> to test original
    model.eval()
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model/processor: {e}")
    exit()

# ----------- LOAD TEST DATA -----------
print(f"Loading test data from: {test_txt_path}")
try:
    test_df = pd.read_csv(test_txt_path, sep="\t", header=None, names=["audio_file", "transcript"])
    test_df = test_df.head(500) # Limiting for testing, remove or adjust as needed
    print(f"Loaded {len(test_df)} test samples.")
except FileNotFoundError:
    print(f"Test data file not found: {test_txt_path}")
    exit()
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

torch.cuda.empty_cache()
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ: # Set only if not already set
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------- SEARCH FUNCTION -----------

def find_audio_file(filename, root_folder):
    """Searches for a .wav file in all subdirectories."""
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            # More robust check: exact match or match with .wav extension
            if file == filename or os.path.splitext(file)[0] == os.path.splitext(filename)[0]:
                # Ensure it's a .wav file if filename itself doesn't specify
                if file.lower().endswith(".wav"):
                    return os.path.join(dirpath, file)
    return None

# ----------- INFERENCE + WER -----------

individual_wers = []
processed_files_count = 0

print("\n🔍 Computing WER for each audio file and saving transcripts:\n")

for i, row in test_df.iterrows():
    file_key = row["audio_file"].strip()
    # The file_key from your TSV might or might not have .wav.
    # We'll primarily use its base name for the output .txt file.
    base_name_for_txt = os.path.splitext(os.path.basename(file_key))[0]

    path = find_audio_file(file_key, search_root)
    ground_truth = str(row["transcript"]) # Ensure ground truth is string

    if path is None or not os.path.exists(path):
        print(f"[{i+1:03d}/{len(test_df):03d}] ❌ File not found for key: {file_key} (Searched in: {search_root})")
        continue

    try:
        # Load and resample audio
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.ndim > 1 and waveform.shape[0] > 1: # If stereo, average channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.ndim == 1: # If mono but not 2D, make it 2D
            waveform = waveform.unsqueeze(0)


        # Preprocess input
        # Whisper expects a 1D numpy array or list of floats
        input_audio_np = waveform.squeeze().numpy()
        inputs = processor(input_audio_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Forced decoding in the target language
        # Ensure 'punjabi' is the correct language code recognized by your fine-tuned model/processor
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="punjabi", task="transcribe")

        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            # Handle cases where prediction might be empty or list of lists
            decoded_predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            predicted_text = decoded_predictions[0] if decoded_predictions else ""
            predicted_text = str(predicted_text) # Ensure predicted text is string

        # Compute WER
        error = wer(ground_truth, predicted_text)
        individual_wers.append(error)
        processed_files_count += 1

        print(f"[{i+1:03d}/{len(test_df):03d}] File: {os.path.basename(path)}")
        print(f"    Ground truth : {ground_truth}")
        print(f"    Prediction   : {predicted_text}")
        print(f"    WER          : {error:.3f}")

        # Save the predicted transcript
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
        # Optionally, append a placeholder or skip WER for this file
        # individual_wers.append(1.0) # Or some other error indicator if you want to count it

# ----------- SUMMARY -----------
if processed_files_count > 0: # Check if any files were successfully processed
    average_wer = sum(individual_wers) / len(individual_wers) # len(individual_wers) should be same as processed_files_count
    print(f"\n📊 Average WER on {processed_files_count} processed samples: {average_wer:.3f}")
    print(f"📂 All successfully predicted transcripts have been saved to: {transcript_output_dir}")
else:
    print("\n📊 No audio files were successfully processed. No WER calculated.")
    if len(test_df) > 0:
         print(f"   Tried to process {len(test_df)} files from the input list.")
    print(f"   Transcript output directory (may be empty): {transcript_output_dir}")