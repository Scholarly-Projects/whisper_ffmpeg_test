import os
import csv
import subprocess
import tempfile
import logging
from pathlib import Path
import numpy as np
import whisper
import torchaudio
import time
import torch
import torchaudio.transforms as T

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODEL_NAME = "small.en"  # Using standard Whisper model
SAMPLE_RATE = 16000  # Required by Whisper
MAX_PAUSE_WITHIN_SEGMENT = 1.0  # Maximum pause within a segment (in seconds)

# Available Whisper models:
# English-only models:
# - tiny.en: ~39M parameters, fastest, least accurate
# - base.en: ~74M parameters, good balance of speed and accuracy
# - small.en: ~244M parameters, better accuracy
# - medium.en: ~769M parameters, even better accuracy
# 
# Multilingual models (support multiple languages):
# - tiny: ~39M parameters, supports 99 languages
# - base: ~74M parameters, supports 99 languages
# - small: ~244M parameters, supports 99 languages
# - medium: ~769M parameters, supports 99 languages
# - large: ~1550M parameters, supports 99 languages, most accurate
# - large-v2: ~1550M parameters, improved version of large model
# - large-v3: ~1550M parameters, latest version with improved accuracy
# - turbo: ~809M parameters, optimized version of large-v3 with faster transcription

def ensure_dir(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path, output_path):
    """Convert audio/video file to WAV format using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_path}: {e.stderr.decode('utf-8')}")
        return False

def transcribe_with_whisper(audio_path, model_name):
    """Transcribe audio using Whisper with word-level timestamps."""
    try:
        # Load Whisper model
        model = whisper.load_model(model_name)
        
        # Transcribe with word-level timestamps
        result = model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            verbose=False
        )
        
        # Extract word segments
        word_segments = []
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                word_segments.append({
                    'start': word.get('start', 0),
                    'end': word.get('end', 0),
                    'word': word.get('word', '').strip()
                })
        
        return word_segments
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {str(e)}")
        return []

def simple_speaker_diarization(word_segments, max_pause=2.0):
    """
    Simple speaker diarization based on pauses and context.
    This is a heuristic approach that alternates speakers when there's a significant pause.
    """
    if not word_segments:
        return []
    
    # Initialize with Speaker 1
    current_speaker = "Speaker 1"
    diarized_segments = []
    
    # Add first segment
    first_segment = word_segments[0].copy()
    first_segment['speaker'] = current_speaker
    diarized_segments.append(first_segment)
    
    for i in range(1, len(word_segments)):
        word = word_segments[i]
        prev_word = word_segments[i-1]
        
        # Calculate pause between words
        pause = word['start'] - prev_word['end']
        
        # If pause is significant, switch speaker
        if pause > max_pause:
            current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"
        
        # Add word with current speaker
        word_with_speaker = word.copy()
        word_with_speaker['speaker'] = current_speaker
        diarized_segments.append(word_with_speaker)
    
    return diarized_segments

def diarize_audio_stereo(audio_path):
    """Perform speaker diarization using stereo channel information."""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Check if stereo
        if waveform.shape[0] < 2:
            logger.warning("Audio is not stereo, using only pause-based diarization")
            return None
        
        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            resampler = T.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Split into left and right channels
        left_channel = waveform[0:1, :]
        right_channel = waveform[1:2, :]
        
        # Simple energy-based voice activity detection
        def detect_voice_activity(channel):
            # Compute energy
            energy = torch.abs(channel)
            
            # Smooth energy
            window_size = int(0.1 * SAMPLE_RATE)  # 100ms window
            if energy.shape[1] < window_size:
                return torch.zeros_like(energy).bool()
            
            # Compute moving average
            kernel = torch.ones(1, 1, window_size) / window_size
            energy = energy.unsqueeze(0).unsqueeze(0)
            smoothed_energy = torch.nn.functional.conv1d(energy, kernel, padding=window_size//2)
            smoothed_energy = smoothed_energy.squeeze()
            
            # Threshold (adjust as needed)
            threshold = smoothed_energy.mean() + 0.5 * smoothed_energy.std()
            
            return smoothed_energy > threshold
        
        # Detect voice activity on each channel
        left_activity = detect_voice_activity(left_channel)
        right_activity = detect_voice_activity(right_channel)
        
        # Create speaker segments
        speaker_segments = []
        
        # Simple approach: assume left channel is speaker 1, right is speaker 2
        # This is a heuristic and may not always be accurate
        
        # Find segments where left channel is active and right is not
        left_only = left_activity & ~right_activity
        # Find segments where right channel is active and left is not
        right_only = right_activity & ~left_activity
        # Find segments where both are active (overlap)
        both_active = left_activity & right_activity
        
        # Convert boolean arrays to segments
        def bool_to_segments(bool_array, speaker):
            segments = []
            start = None
            for i, active in enumerate(bool_array):
                if active and start is None:
                    start = i / SAMPLE_RATE
                elif not active and start is not None:
                    end = i / SAMPLE_RATE
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })
                    start = None
            return segments
        
        # Get segments for each case
        left_segments = bool_to_segments(left_only, "Speaker 1")
        right_segments = bool_to_segments(right_only, "Speaker 2")
        both_segments = bool_to_segments(both_active, "Speaker 1")  # Default to speaker 1 for overlaps
        
        # Combine all segments
        speaker_segments = left_segments + right_segments + both_segments
        
        # Sort by start time
        speaker_segments.sort(key=lambda x: x['start'])
        
        return speaker_segments
    except Exception as e:
        logger.error(f"Error in stereo diarization: {str(e)}")
        return None

def assign_speakers_to_words(word_segments, speaker_segments):
    """Assign speakers to word segments based on overlap."""
    assigned_words = []
    
    for word in word_segments:
        word_start = word['start']
        word_end = word['end']
        
        # Find speaker segment with maximum overlap
        max_overlap = 0
        assigned_speaker = "Speaker 1"  # Default
        
        if speaker_segments:
            for segment in speaker_segments:
                seg_start = segment['start']
                seg_end = segment['end']
                
                # Calculate overlap
                overlap_start = max(word_start, seg_start)
                overlap_end = min(word_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = segment['speaker']
        
        word['speaker'] = assigned_speaker
        assigned_words.append(word)
    
    return assigned_words

def format_time(seconds):
    """Format time in seconds to HH:MM:SS:XX format for timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    hundredths = int((seconds - int(seconds)) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{hundredths:02d}"

def format_end_time(seconds):
    """Format time in seconds to [HH:MM:SS:XX] format for End Time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    hundredths = int((seconds - int(seconds)) * 100)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}:{hundredths:02d}]"

def group_segments(word_segments, max_pause=MAX_PAUSE_WITHIN_SEGMENT):
    """Group word segments by speaker and pause."""
    if not word_segments:
        return []
    
    segments = []
    current_segment = {
        'speaker': word_segments[0]['speaker'],
        'start': word_segments[0]['start'],
        'end': word_segments[0]['end'],
        'words': [word_segments[0]['word']]
    }
    
    for i in range(1, len(word_segments)):
        word = word_segments[i]
        prev_word = word_segments[i-1]
        
        # Check if we should start a new segment
        pause = word['start'] - prev_word['end']
        speaker_change = word['speaker'] != current_segment['speaker']
        
        if speaker_change or pause > max_pause:
            # Save current segment
            segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                'speaker': word['speaker'],
                'start': word['start'],
                'end': word['end'],
                'words': [word['word']]
            }
        else:
            # Add to current segment
            current_segment['end'] = word['end']
            current_segment['words'].append(word['word'])
    
    # Add the last segment
    segments.append(current_segment)
    
    return segments

def process_file(input_path, output_csv_path):
    """Process a single audio/video file."""
    logger.info(f"Processing {input_path}")
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    try:
        # Convert to WAV
        if not convert_to_wav(input_path, temp_wav_path):
            return False
        
        # Transcribe with Whisper
        word_segments = transcribe_with_whisper(temp_wav_path, MODEL_NAME)
        if not word_segments:
            logger.warning(f"No words transcribed in {input_path}")
            return False
        
        # Try stereo diarization first
        speaker_segments = diarize_audio_stereo(temp_wav_path)
        
        if speaker_segments:
            # Use stereo-based speaker assignment
            assigned_words = assign_speakers_to_words(word_segments, speaker_segments)
        else:
            # Fall back to pause-based diarization
            logger.info("Using pause-based speaker diarization")
            assigned_words = simple_speaker_diarization(word_segments)
        
        # Group words into segments
        segments = group_segments(assigned_words)
        
        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['speaker', 'timestamp', 'End Time', 'words']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for segment in segments:
                # Join words with spaces
                text = ' '.join(segment['words'])
                
                writer.writerow({
                    'speaker': segment['speaker'],
                    'timestamp': f"[{format_time(segment['start'])}]",
                    'End Time': format_end_time(segment['end']),
                    'words': text
                })
        
        logger.info(f"Created CSV: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

def main():
    start_time = time.time()
    
    # Ensure directories exist
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    
    # Find input files
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    supported_extensions = audio_extensions + video_extensions
    
    input_files = []
    for ext in supported_extensions:
        input_files.extend(Path(INPUT_DIR).glob(f"*{ext}"))
    
    if not input_files:
        logger.error(f"No supported audio/video files found in {INPUT_DIR}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    logger.info(f"Using Whisper model: {MODEL_NAME}")
    logger.info(f"Speaker diarization: Pause-based + Stereo Diarization")
    
    # Process each file
    success_count = 0
    for input_path in input_files:
        output_csv_path = Path(OUTPUT_DIR) / f"{input_path.stem}.csv"
        
        if process_file(str(input_path), str(output_csv_path)):
            success_count += 1
    
    logger.info(f"Processed {success_count}/{len(input_files)} files successfully")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()