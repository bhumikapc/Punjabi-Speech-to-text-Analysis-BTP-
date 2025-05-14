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

# Check CPU Info
try:
    info = cpuinfo.get_cpu_info()
    logger.info(f"Successfully retrieved CPU info: {info.get('brand_raw', 'N/A')}")
except Exception as e:
    logger.warning(f"Could not get CPU info using cpuinfo library: {e}")

# Model Configuration
WHISPER_MODEL_NAME = "DrishtiSharma/whisper-large-v2-punjabi-700-steps"
LANGUAGE = "punjabi"
TASK = "transcribe"

# Environment Configuration
torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    #login(token="____") #--> enter your login token
    logger.info("Hugging Face login successful.")
except Exception as e:
    logger.warning(f"Hugging Face login failed (using cached token if available): {e}")


# Paths
BASE_DIR = os.getenv("BTP_DIR", "/home/bhumika/BTP")
police_files_path = os.path.join(BASE_DIR, "whisper_ash/police_files/Low/")
output_folder = os.path.join(BASE_DIR, 'whisper_ash/police_files/predicted_transcripts_original/LOW_sentences2')

# Device Setup 
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    fp16_enabled = True # Enable FP16 for faster inference on compatible GPUs
else:
    device = torch.device("cpu")
    logger.info("Using CPU device.")
    fp16_enabled = False # FP16 not typically beneficial on CPU


logger.info(f"Loading Whisper processor for model: {WHISPER_MODEL_NAME}")
try:
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}")

    # Load the model in default precision (float32) first
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_NAME
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
    """Loads an audio file and resamples it to 16kHz if needed."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
             waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)
        waveform = waveform.to(torch.float32)
        return waveform.squeeze(0) # Return 1D tensor [samples]
    except Exception as e:
        logger.error(f"Error loading or resampling audio file {file_path}: {e}", exc_info=True)
        return None

# transcribe the audio
def transcribe_audio_whisper(model, processor, audio_waveform, language=LANGUAGE, task=TASK):
    """Transcribes a single audio waveform using the Whisper model."""
    if audio_waveform is None: return "Error: Could not load audio."
    if audio_waveform.nelement() == 0: return "Error: Empty audio waveform."

    duration_seconds = audio_waveform.shape[0] / 16000.0
    logger.debug(f"Input audio duration: {duration_seconds:.2f} seconds")

    if torch.max(torch.abs(audio_waveform)) < 1e-4:
        logger.warning(f"Input audio waveform appears to be silent or near-silent (duration: {duration_seconds:.2f}s).")
        return "<silent_audio>"

    try:
        input_features = processor(
            audio_waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
    except Exception as e:
         logger.error(f"Error processing audio features: {e}", exc_info=True)
         return "Error: Could not process audio features."

    input_features = input_features.to(device)
    if model.dtype == torch.float16:
        input_features = input_features.half()

    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        prompt_token_length = len(forced_decoder_ids[0]) if forced_decoder_ids else 0
        logger.debug(f"Forced decoder IDs length: {prompt_token_length}")
        logger.debug(f"Forced decoder IDs: {forced_decoder_ids}")
    except Exception as e:
        logger.error(f"Error getting decoder prompt IDs: {e}", exc_info=True)
        return "Error: Failed to get decoder prompt IDs."

    adjusted_max_new_tokens = 444

    with torch.no_grad():
        try:
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=2,
                repetition_penalty=1.1,
                max_new_tokens=adjusted_max_new_tokens, 
                early_stopping=False
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
                logger.warning("CUDA OutOfMemoryError caught. Cleared cache. Transcription for this file may fail.")
            return "Error: Model inference failed."

    try:
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.debug(f"Decoded transcription (special tokens skipped): '{transcription}'")

        raw_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        logger.debug(f"Raw transcription (includes special tokens): '{raw_transcription}'")
        if "<|endoftext|>" in raw_transcription and len(raw_transcription) < 50 :
             logger.warning("Potential premature EOS detected in raw output.")

        if not transcription or transcription.isspace():
             logger.warning("Transcription result is empty or whitespace.")
             return ""

        return transcription.strip()
    except Exception as e:
        logger.error(f"Error decoding predicted IDs: {e}", exc_info=True)
        return "Error: Decoding failed."

# Main Loop for generating Transcriptions
logger.info("Starting transcription process...")
transcriptions = []

for i, file_path in enumerate(police_files):
    abs_file_path = os.path.abspath(file_path)
    logger.info(f"Processing file {i+1}/{len(police_files)}: {os.path.basename(abs_file_path)}")

    audio_waveform = load_audio(abs_file_path)
    transcription_text = "Error: Initial value" # Default value
    if audio_waveform is None:
        transcription_text = "Error: Could not load audio."
    elif audio_waveform.nelement() == 0:
        transcription_text = "Error: Loaded audio is empty."
    else:
        transcription_text = transcribe_audio_whisper(
            model, processor, audio_waveform, language=LANGUAGE, task=TASK
        )

    log_preview = transcription_text[:100].replace('\n', ' ') + ('...' if len(transcription_text) > 100 else '')
    logger.info(f"  -> Transcription: {log_preview}")
    transcriptions.append((abs_file_path, transcription_text))

    if (i + 1) % 10 == 0 and device == torch.device("cuda"):
         torch.cuda.empty_cache()
         logger.debug("Cleared CUDA cache periodically.")


# Save Transcriptions
logger.info(f"Saving transcriptions to: {output_folder}")
os.makedirs(output_folder, exist_ok=True)

saved_count = 0
error_count = 0
empty_count = 0
for abs_file_path, transcription in transcriptions:
    district = file_district_map.get(abs_file_path, "UnknownDistrict")
    audio_file_name = os.path.splitext(os.path.basename(abs_file_path))[0]
    txt_file_name = f"Low_{district}_Punjabi_{audio_file_name}.txt"
    txt_file_path = os.path.join(output_folder, txt_file_name)

    try:
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        saved_count += 1
        if transcription.startswith("Error:"):
            error_count += 1
            logger.warning(f"Saved transcription with error message for {os.path.basename(abs_file_path)}: {transcription}")
        elif not transcription or transcription.isspace() or transcription == "<silent_audio>": # Also count silence as 'empty' for summary
            empty_count +=1
            if transcription == "<silent_audio>":
                 logger.warning(f"Saved SILENT audio marker for {os.path.basename(abs_file_path)} to {txt_file_path}")
            else:
                 logger.warning(f"Saved EMPTY transcription for {os.path.basename(abs_file_path)} to {txt_file_path}")
    except Exception as e:
        logger.error(f"Failed to save transcript for {os.path.basename(abs_file_path)} to {txt_file_path}: {e}", exc_info=True)
        error_count += 1

logger.info(f"Finished processing.")
logger.info(f"  Total files processed: {len(police_files)}")
logger.info(f"  Transcriptions saved: {saved_count}")
logger.info(f"  Transcriptions indicating errors (loading/processing/saving): {error_count}")
logger.info(f"  Transcriptions that were empty/whitespace/silent: {empty_count}")
logger.info(f"Output folder: {output_folder}")

