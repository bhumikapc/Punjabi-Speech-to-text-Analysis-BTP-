# -*- coding: utf-8 -*-
"""
Script to transcribe police audio files using a Punjabi Whisper model.
"""

import torchaudio
import pandas as pd
import numpy as np
import torch
import os
from glob import glob
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login
import logging
import cpuinfo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Check CPU Info
try:
    info = cpuinfo.get_cpu_info()
    logger.info(f"Successfully retrieved CPU info: {info.get('brand_raw', 'N/A')}")
except Exception as e:
    logger.warning(f"Could not get CPU info using cpuinfo library: {e}")

# Model Configuration
WHISPER_MODEL_PATH = '/home/bhumika/BTP/whisper_ash/my_finetuned_model' 
LANGUAGE = "punjabi"
TASK = "transcribe"

# Audio Processing Configuration
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 30.0
CHUNK_SAMPLES = int(CHUNK_DURATION_SECONDS * TARGET_SAMPLE_RATE)


# Environment Configuration
torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Keep commented for now

# hugging face login
try:
    #login(token="____") #--> enter your login token
    logger.info("Hugging Face login successful.")
except Exception as e:
    logger.warning(f"Hugging Face login failed (using cached token if available): {e}")


# Paths
BASE_DIR = os.getenv("BTP_DIR", "/home/bhumika/BTP") # Or adjust the default path as needed
police_files_path = os.path.join(BASE_DIR, "whisper_ash/police_files/Low/")
output_folder = os.path.join(BASE_DIR, 'whisper_ash/police_files/predicted_transcripts_finetuned/LOW_sentences2')

# Device Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    fp16_enabled = True # Enable FP16 for faster inference on compatible GPUs
else:
    device = torch.device("cpu")
    logger.info("Using CPU device.")
    fp16_enabled = False # FP16 not typically beneficial on CPU

logger.info(f"Loading Whisper processor from local path: {WHISPER_MODEL_PATH}")
try:
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    logger.info(f"Loading Whisper model from local path: {WHISPER_MODEL_PATH}")

    # Load the model in default precision (float32) first
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_PATH
    )


    # Move the model to the target device
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # If using CUDA and FP16 is enabled, convert the model to half precision
    if fp16_enabled and device == torch.device("cuda"):
        model.half()
        logger.info("Model converted to FP16 for CUDA inference.")

    model.eval() # Set model to evaluation mode
    logger.info("Whisper model and processor loaded successfully.")

except Exception as e:
    if "Failed to initialize cpuinfo!" in str(e):
         logger.error(f"Error loading Whisper model or processor: {e}", exc_info=True)
         logger.error("This specific error often relates to the 'py-cpuinfo' package.")
         logger.error("Try reinstalling it: 'pip uninstall py-cpuinfo && pip install py-cpuinfo'")
         logger.error("If the error persists, check system permissions for /proc/cpuinfo or environment issues.")
    else:
         logger.error(f"Error loading Whisper model or processor: {e}", exc_info=True) # Log full traceback
         logger.error("Please check model name, internet connection, HF credentials, and available VRAM.")
    exit(1)


# Load Police Audio File Paths
districts = ['Kapurthala', 'Amritsar', 'Ropar', 'Patiala']
police_files = []
file_district_map = {} 

logger.info(f"Scanning for audio files in: {police_files_path}")
for district in districts:
    punjabi_folder = os.path.join(police_files_path, district, 'Punjabi')
    wav_files = glob(os.path.join(punjabi_folder, '*.wav')) + glob(os.path.join(punjabi_folder, '*.WAV'))
    logger.info(f"District: {district}, Files found: {len(wav_files)}")
    police_files.extend(wav_files)
    for file_path in wav_files:
        abs_file_path = os.path.abspath(file_path)
        file_district_map[abs_file_path] = district

logger.info(f"Total audio files found: {len(police_files)}")
if not police_files:
    logger.warning("No audio files found. Exiting.")
    exit(0)


# Load the audio and resample to 16kHz
def load_audio(file_path):
    """Loads an audio file and resamples it to TARGET_SAMPLE_RATE if needed."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
             waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != TARGET_SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = transform(waveform)
        waveform = waveform.to(torch.float32)
        return waveform.squeeze(0) # Return 1D tensor [samples]
    except Exception as e:
        logger.error(f"Error loading or resampling audio file {file_path}: {e}", exc_info=True)
        return None

# transcribe the audio
def transcribe_audio_whisper(model, processor, audio_waveform, language=LANGUAGE, task=TASK):
    """Transcribes a single audio waveform (or chunk) using the Whisper model."""
    if audio_waveform is None: return "Error: Could not load audio." # Should be caught earlier
    if audio_waveform.nelement() == 0: return "Error: Empty audio waveform."

    duration_seconds = audio_waveform.shape[0] / float(TARGET_SAMPLE_RATE)
    logger.debug(f"Input audio (or chunk) duration: {duration_seconds:.2f} seconds")

    if torch.max(torch.abs(audio_waveform)) < 1e-4: # Check for near-silence
        logger.warning(f"Input audio (or chunk) appears to be silent or near-silent (duration: {duration_seconds:.2f}s).")
        return "<silent_audio>"

    try:
        input_features = processor(
            audio_waveform.numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt"
        ).input_features
    except Exception as e:
         logger.error(f"Error processing audio features: {e}", exc_info=True)
         return "Error: Could not process audio features."

    input_features = input_features.to(device)
    if model.dtype == torch.float16: # If fp16 is enabled for the model
        input_features = input_features.half()

    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        prompt_token_length = len(forced_decoder_ids[0]) if forced_decoder_ids else 0
        logger.debug(f"Forced decoder IDs length: {prompt_token_length}")
        logger.debug(f"Forced decoder IDs: {forced_decoder_ids}")
    except Exception as e:
        logger.error(f"Error getting decoder prompt IDs: {e}", exc_info=True)
        return "Error: Failed to get decoder prompt IDs."

    adjusted_max_new_tokens = 444 # Max tokens model can generate for the content itself

    with torch.no_grad():
        try:
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=5,
                repetition_penalty=1.1,
                max_new_tokens=adjusted_max_new_tokens,
                early_stopping=False # As per original user code
            )
            logger.debug(f"Raw predicted IDs shape: {predicted_ids.shape}")
        except ValueError as ve:
             logger.error(f"ValueError during model generation: {ve}", exc_info=True)
             logger.error(f"Check prompt length ({prompt_token_length}) vs max_new_tokens ({adjusted_max_new_tokens}) vs model limit ({model.config.max_length}).")
             return "Error: Model generation configuration error."
        except Exception as e:
            logger.error(f"Error during model generation (inference): {e}", exc_info=True)
            if "out of memory" in str(e).lower() and device == torch.device("cuda"):
                torch.cuda.empty_cache()
                logger.warning("CUDA OutOfMemoryError caught. Cleared cache. Transcription for this segment may fail.")
            return "Error: Model inference failed."

    try:
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.debug(f"Decoded transcription (special tokens skipped): '{transcription}'")

        raw_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        logger.debug(f"Raw transcription (includes special tokens): '{raw_transcription}'")
        if "<|endoftext|>" in raw_transcription and len(raw_transcription) < 50 : # Heuristic
             logger.warning("Potential premature EOS detected in raw output for segment.")

        if not transcription or transcription.isspace():
             logger.warning("Transcription result for segment is empty or whitespace.")
             return "" # Return empty string for consistent joining later

        return transcription.strip()
    except Exception as e:
        logger.error(f"Error decoding predicted IDs: {e}", exc_info=True)
        return "Error: Decoding failed."

# Main Transcription Loop
logger.info("Starting transcription process...")
transcriptions = []

for i, file_path in enumerate(police_files):
    abs_file_path = os.path.abspath(file_path)
    base_audio_filename = os.path.basename(abs_file_path)
    logger.info(f"Processing file {i+1}/{len(police_files)}: {base_audio_filename}")

    audio_waveform = load_audio(abs_file_path)
    transcription_text = "Error: Initial value" 

    if audio_waveform is None:
        transcription_text = "Error: Could not load audio."
    elif audio_waveform.nelement() == 0:
        transcription_text = "Error: Loaded audio is empty."
    else:
        file_duration_seconds = audio_waveform.shape[0] / float(TARGET_SAMPLE_RATE)
        logger.info(f"  Audio file duration: {file_duration_seconds:.2f} seconds for {base_audio_filename}")

        if file_duration_seconds <= CHUNK_DURATION_SECONDS:
            logger.info(f"  Audio is <= {CHUNK_DURATION_SECONDS}s, processing as a single segment.")
            transcription_text = transcribe_audio_whisper(
                model, processor, audio_waveform, language=LANGUAGE, task=TASK
            )
        else:
            logger.info(f"  Audio is > {CHUNK_DURATION_SECONDS}s, processing in {CHUNK_DURATION_SECONDS}s chunks.")
            num_chunks = int(np.ceil(file_duration_seconds / CHUNK_DURATION_SECONDS))
            logger.info(f"  Splitting {base_audio_filename} into {num_chunks} chunks.")
            
            all_chunk_transcriptions = []
            for chunk_idx in range(num_chunks):
                start_sample = chunk_idx * CHUNK_SAMPLES
                end_sample = min((chunk_idx + 1) * CHUNK_SAMPLES, audio_waveform.shape[0])
                current_chunk_waveform = audio_waveform[start_sample:end_sample]

                if current_chunk_waveform.nelement() == 0:
                    logger.warning(f"    Chunk {chunk_idx + 1}/{num_chunks} for {base_audio_filename} is empty, skipping.")
                    all_chunk_transcriptions.append("") # Append empty string to maintain order if needed, or skip
                    continue
                
                chunk_duration = current_chunk_waveform.shape[0] / float(TARGET_SAMPLE_RATE)
                logger.info(f"    Transcribing chunk {chunk_idx + 1}/{num_chunks} (Duration: {chunk_duration:.2f}s, Samples: {start_sample}-{end_sample}) for {base_audio_filename}")
                
                chunk_transcription = transcribe_audio_whisper(
                    model, processor, current_chunk_waveform, language=LANGUAGE, task=TASK
                )
                all_chunk_transcriptions.append(chunk_transcription)
                

            # Join transcriptions from all chunks
            # Filter out completely empty strings from the join, but keep errors and <silent_audio>
            transcription_text = " ".join(t for t in all_chunk_transcriptions if t).strip() # `if t` filters None and ""
            
            if not transcription_text: # If all chunks resulted in errors that were filtered or empty strings
                # Check if all_chunk_transcriptions contained only errors or were all empty
                if all(t.startswith("Error:") or not t or t.isspace() for t in all_chunk_transcriptions):
                    transcription_text = f"Error: All {num_chunks} chunks resulted in errors or were empty for {base_audio_filename}."
                else: 
                     transcription_text = f"Error: Transcription from {num_chunks} chunks resulted in empty text for {base_audio_filename}."

            logger.info(f"  Combined transcription from {num_chunks} chunks for {base_audio_filename}.")
            
            # Clear CUDA cache after processing all chunks of a single (long) file
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
                logger.debug(f"  Cleared CUDA cache after processing all chunks for {base_audio_filename}.")

    log_preview = transcription_text[:100].replace('\n', ' ') + ('...' if len(transcription_text) > 100 else '')
    logger.info(f"  -> Final Transcription for {base_audio_filename}: {log_preview}")
    transcriptions.append((abs_file_path, transcription_text))

    # Original periodic cache clearing (after processing a certain number of files)
    if (i + 1) % 10 == 0 and device == torch.device("cuda"):
         torch.cuda.empty_cache()
         logger.debug(f"Cleared CUDA cache periodically after processing file {i+1}.")


# Save Transcriptions
logger.info(f"Saving transcriptions to: {output_folder}")
os.makedirs(output_folder, exist_ok=True)

saved_count = 0
error_count = 0 # Counts files where the final transcription indicates an error
empty_count = 0 # Counts files where the final transcription is empty or just <silent_audio>

for abs_file_path, transcription in transcriptions:
    district = file_district_map.get(abs_file_path, "UnknownDistrict")
    audio_file_name_base = os.path.splitext(os.path.basename(abs_file_path))[0]
    txt_file_name = f"Low_{district}_Punjabi_{audio_file_name_base}.txt"
    txt_file_path = os.path.join(output_folder, txt_file_name)

    try:
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        saved_count += 1
        
        # Refined counting for summary
        is_error_transcription = transcription.startswith("Error:")
        is_empty_or_silent = not transcription or transcription.isspace() or transcription == "<silent_audio>" or \
                             (transcription.count("<silent_audio>") > 0 and \
                              len(transcription.replace("<silent_audio>", "").replace(" ", "")) == 0) # Handles multiple silent chunks

        if is_error_transcription:
            error_count += 1
            logger.warning(f"Saved transcription with error message for {os.path.basename(abs_file_path)}: {transcription[:150]}...")
        elif is_empty_or_silent:
            empty_count +=1
            if "<silent_audio>" in transcription:
                 logger.warning(f"Saved SILENT audio marker(s) for {os.path.basename(abs_file_path)} to {txt_file_path}")
            else:
                 logger.warning(f"Saved EMPTY transcription for {os.path.basename(abs_file_path)} to {txt_file_path}")
                 
    except Exception as e:
        logger.error(f"Failed to save transcript for {os.path.basename(abs_file_path)} to {txt_file_path}: {e}", exc_info=True)
        if not transcription.startswith("Error:"): # If transcription was okay, but saving failed
            error_count += 1 # Count it as an error if not already an error transcription

logger.info(f"Finished processing.")
logger.info(f"  Total audio files found: {len(police_files)}")
logger.info(f"  Total transcriptions generated (attempted): {len(transcriptions)}")
logger.info(f"  Transcriptions successfully saved to .txt files: {saved_count}")
logger.info(f"  Files resulting in error transcriptions (e.g., 'Error: ...'): {error_count}")
logger.info(f"  Files resulting in empty/whitespace or purely silent transcriptions: {empty_count}")
logger.info(f"Output folder: {output_folder}")