import os
import csv
import subprocess
import tempfile
import logging
from pathlib import Path
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils import get_device
import whisper_cpp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODEL_PATH = "base.en.bin"  # Download this from whisper.cpp models
SAMPLE_RATE = 16000  # Required by whisper.cpp

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

def transcribe_with_whisper(audio_path, model_path):
    """Transcribe audio using whisper.cpp with word-level timestamps."""
    try:
        # Load whisper.cpp model
        model = whisper_cpp.Whisper(model_path)
        
        # Transcribe with word-level timestamps
        result = model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            print_progress=False
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

def diarize_audio(audio_path):
    """Perform speaker diarization using pyannote.audio."""
    try:
        # Initialize diarization pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=False  # Set to True if you have Hugging Face token
        )
        
        # Send pipeline to appropriate device
        device = get_device()
        pipeline.to(device)
        
        # Process audio file
        diarization = pipeline(audio_path)
        
        # Extract speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        return speaker_segments
    except Exception as e:
        logger.error(f"Error in diarization: {str(e)}")
        return []

def assign_speakers_to_words(word_segments, speaker_segments):
    """Assign speakers to word segments based on overlap."""
    assigned_words = []
    
    for word in word_segments:
        word_start = word['start']
        word_end = word['end']
        
        # Find speaker segment with maximum overlap
        max_overlap = 0
        assigned_speaker = "Unknown"
        
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
        
        # Convert speaker label to "Speaker 1", "Speaker 2", etc.
        if assigned_speaker != "Unknown":
            try:
                speaker_num = int(assigned_speaker.split('_')[-1]) + 1
                assigned_speaker = f"Speaker {speaker_num}"
            except (ValueError, IndexError):
                assigned_speaker = "Unknown"
        
        assigned_words.append({
            'speaker': assigned_speaker,
            'start': word_start,
            'end': word_end,
            'word': word['word']
        })
    
    return assigned_words

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
        
        # Transcribe with whisper.cpp
        word_segments = transcribe_with_whisper(temp_wav_path, MODEL_PATH)
        if not word_segments:
            logger.warning(f"No words transcribed in {input_path}")
            return False
        
        # Perform speaker diarization
        speaker_segments = diarize_audio(temp_wav_path)
        if not speaker_segments:
            logger.warning(f"No speaker segments found in {input_path}")
            # Assign all words to "Speaker 1" as fallback
            assigned_words = [{
                'speaker': 'Speaker 1',
                'start': word['start'],
                'end': word['end'],
                'word': word['word']
            } for word in word_segments]
        else:
            # Assign speakers to words
            assigned_words = assign_speakers_to_words(word_segments, speaker_segments)
        
        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['speaker', 'time start', 'time stop', 'words']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for word_data in assigned_words:
                writer.writerow({
                    'speaker': word_data['speaker'],
                    'time start': f"{word_data['start']:.2f}",
                    'time stop': f"{word_data['end']:.2f}",
                    'words': word_data['word']
                })
        
        logger.info(f"Created CSV: {output_csv_path}")
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
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Download models from: https://github.com/ggerganov/whisper.cpp/tree/master/models")
        return
    
    # Supported file extensions
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    supported_extensions = audio_extensions + video_extensions
    
    # Find input files
    input_files = []
    for ext in supported_extensions:
        input_files.extend(Path(INPUT_DIR).glob(f"*{ext}"))
    
    if not input_files:
        logger.error(f"No supported audio/video files found in {INPUT_DIR}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    success_count = 0
    for input_path in input_files:
        output_csv_path = Path(OUTPUT_DIR) / f"{input_path.stem}.csv"
        
        if process_file(str(input_path), str(output_csv_path)):
            success_count += 1
    
    logger.info(f"Processed {success_count}/{len(input_files)} files successfully")
    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    import time
    main()