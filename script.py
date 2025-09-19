import os
import csv
import subprocess
import tempfile
import logging
from pathlib import Path
import whisper
import torch
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import KMeans
import numpy as np
import torchaudio
import torchaudio.transforms as T
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODEL_NAME = "small.en"
SAMPLE_RATE = 16000
MAX_SPEAKERS = 4

# Initialize models
logger.info("Loading models...")
whisper_model = whisper.load_model(MODEL_NAME)
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
)


def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def convert_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-i", input_path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_path}: {e}")
        return False


def extract_segment_embedding(audio_path, start_time, end_time):
    """Extract speaker embedding for a specific time segment."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int(end_time * SAMPLE_RATE)
        
        if end_sample > waveform.size(1):
            end_sample = waveform.size(1)
        if start_sample >= end_sample:
            return None
            
        segment_wave = waveform[:, start_sample:end_sample]
        segment_wave = segment_wave / (segment_wave.abs().max() + 1e-8)
        
        with torch.no_grad():
            embedding = speaker_encoder.encode_batch(segment_wave)
            return embedding.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning(f"Failed to extract embedding for segment {start_time}-{end_time}: {e}")
        return None


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    hundredths = int((seconds - int(seconds)) * 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{hundredths:02d}"


def format_end_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    hundredths = int((seconds - int(seconds)) * 100)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}:{hundredths:02d}]"


def process_file(input_path, output_csv_path):
    logger.info(f"Processing {input_path}")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    try:
        # Convert to WAV
        if not convert_to_wav(input_path, temp_wav_path):
            return False
        
        # Transcribe with Whisper (let it do the heavy lifting)
        logger.info("Transcribing with Whisper...")
        result = whisper_model.transcribe(
            temp_wav_path,
            language="en",
            word_timestamps=True,
            verbose=False
        )
        
        # Extract segments from Whisper result
        segments = result.get('segments', [])
        if not segments:
            logger.warning("No segments found in transcription")
            return False
        
        # Extract embeddings for each segment
        embeddings = []
        valid_segments = []
        
        for segment in segments:
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Only process segments longer than 1 second
            if end_time - start_time < 1.0:
                continue
                
            embedding = extract_segment_embedding(temp_wav_path, start_time, end_time)
            if embedding is not None:
                embeddings.append(embedding)
                valid_segments.append(segment)
        
        if len(embeddings) == 0:
            # Fallback: assign all to Speaker 1
            logger.warning("No valid embeddings, assigning all to Speaker 1")
            speaker_labels = [0] * len(segments)
            segments_to_use = segments
        else:
            # Simple K-means clustering on embeddings
            embeddings = np.array(embeddings)
            n_clusters = min(MAX_SPEAKERS, len(embeddings))
            
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
            else:
                cluster_labels = [0] * len(embeddings)
            segments_to_use = valid_segments
        
        # Map cluster labels to speaker names based on order of appearance
        # First, create a mapping from cluster label to speaker number in order of appearance
        speaker_counter = 1
        cluster_to_speaker = {}
        speaker_labels = []
        
        for i, label in enumerate(cluster_labels):
            if label not in cluster_to_speaker:
                cluster_to_speaker[label] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            speaker_labels.append(cluster_to_speaker[label])
        
        # Assign speakers to segments and group consecutive segments from same speaker
        temp_segments = []
        for i, segment in enumerate(segments_to_use):
            speaker = speaker_labels[i]
            temp_segments.append({
                'speaker': speaker,
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '').strip()
            })
        
        # Group consecutive segments from the same speaker
        final_segments = []
        if temp_segments:
            current_segment = {
                'speaker': temp_segments[0]['speaker'],
                'start': temp_segments[0]['start'],
                'end': temp_segments[0]['end'],
                'text': temp_segments[0]['text']
            }
            
            for segment in temp_segments[1:]:
                # If same speaker, merge regardless of pause duration
                if segment['speaker'] == current_segment['speaker']:
                    current_segment['end'] = segment['end']
                    current_segment['text'] += ' ' + segment['text']
                else:
                    # Different speaker, start new segment
                    final_segments.append(current_segment)
                    current_segment = {
                        'speaker': segment['speaker'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text']
                    }
            
            # Don't forget the last segment
            final_segments.append(current_segment)
        
        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['speaker', 'timestamp', 'End Time', 'words']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for segment in final_segments:
                writer.writerow({
                    'speaker': segment['speaker'],
                    'timestamp': f"[{format_time(segment['start'])}]",
                    'End Time': format_end_time(segment['end']),
                    'words': segment['text']
                })
        
        logger.info(f"Created CSV: {output_csv_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)


def main():
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    
    # Find supported files
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    supported_extensions = audio_extensions + video_extensions
    
    input_files = []
    for ext in supported_extensions:
        input_files.extend(Path(INPUT_DIR).glob(f"*{ext}"))
    
    if not input_files:
        logger.error(f"No supported files found in {INPUT_DIR}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    success_count = 0
    for input_path in input_files:
        output_csv_path = Path(OUTPUT_DIR) / f"{input_path.stem}.csv"
        if process_file(str(input_path), str(output_csv_path)):
            success_count += 1
    
    logger.info(f"Processed {success_count}/{len(input_files)} files successfully")


if __name__ == "__main__":
    main()
