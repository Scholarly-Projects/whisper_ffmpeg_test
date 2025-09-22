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
from collections import defaultdict

# === CONFIG ===
INPUT_DIR = "A"
OUTPUT_DIR = "B"
MODEL_NAME = "small.en"  # Use "base.en" for faster, "medium.en" for more accurate
SAMPLE_RATE = 16000
MAX_SPEAKERS = 4

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === MODELS ===
logger.info("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(MODEL_NAME, device=device)
speaker_encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb",
    run_opts={"device": device}
)

# === HELPERS ===

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path, output_path):
    cmd = ["ffmpeg", "-i", input_path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed on {input_path}: {e}")
        return False

def extract_embedding(audio_path, start, end):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        start_samp = int(start * SAMPLE_RATE)
        end_samp = int(end * SAMPLE_RATE)
        if end_samp > waveform.size(1): end_samp = waveform.size(1)
        if start_samp >= end_samp: return None
        segment = waveform[:, start_samp:end_samp]
        segment = segment / (segment.abs().max() + 1e-8)  # normalize
        with torch.no_grad():
            emb = speaker_encoder.encode_batch(segment)
            return emb.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding failed for [{start:.1f}-{end:.1f}s]: {e}")
        return None

def is_sentence_end(text):
    return text.strip().endswith(('.', '?', '!', '."', '?"', '!"'))

def format_timestamp(seconds, include_brackets=False):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    ts = f"{h:02d}:{m:02d}:{s:02d}:{ms:02d}"
    return f"[{ts}]" if include_brackets else ts

# === MAIN PROCESSING ===

def process_file(input_path, output_csv_path):
    logger.info(f"Processing: {input_path.name}")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name

    try:
        if not convert_to_wav(str(input_path), wav_path):
            return False

        # Transcribe with word timestamps for segmentation
        result = whisper_model.transcribe(
            wav_path,
            language="en",
            word_timestamps=True,
            verbose=False
        )

        segments = result.get('segments', [])
        if not segments:
            logger.warning("No segments transcribed.")
            return False

        # Group into sentence-level segments (safe boundaries)
        sentence_segments = []
        current = {'start': None, 'text': '', 'words': []}

        for segment in segments:
            words = segment.get('words', [])
            for word in words:
                if current['start'] is None:
                    current['start'] = word['start']
                current['text'] += word['word']
                current['words'].append(word)
                if is_sentence_end(current['text']):
                    current['end'] = word['end']
                    sentence_segments.append(current.copy())
                    current = {'start': None, 'text': '', 'words': []}
            # Flush remaining if segment ends
            if current['text'] and (not words or segment == segments[-1]):
                current['end'] = segment['end']
                sentence_segments.append(current)
                current = {'start': None, 'text': '', 'words': []}

        if not sentence_segments:
            logger.warning("No sentence segments created. Using raw segments.")
            sentence_segments = [
                {'start': s['start'], 'end': s['end'], 'text': s['text']}
                for s in segments
            ]

        # Extract embeddings (min 1.5s for stability)
        embeddings = []
        valid_segments = []

        for seg in sentence_segments:
            dur = seg['end'] - seg['start']
            if dur < 1.5:
                continue
            emb = extract_embedding(wav_path, seg['start'], seg['end'])
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append(seg)

        # Assign speakers
        speaker_labels = []
        if len(embeddings) < 2:
            # Not enough to cluster → assign all to Speaker 1
            speaker_labels = ["Speaker 1"] * len(sentence_segments)
        else:
            # Cluster valid segments
            n_clusters = min(MAX_SPEAKERS, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(np.array(embeddings))
            cluster_to_speaker = {}
            speaker_id = 1
            for cid in cluster_ids:
                if cid not in cluster_to_speaker:
                    cluster_to_speaker[cid] = f"Speaker {speaker_id}"
                    speaker_id += 1
            # Map clustered segments
            speaker_map = {}
            for i, seg in enumerate(valid_segments):
                speaker_map[(seg['start'], seg['end'])] = cluster_to_speaker[cluster_ids[i]]
            # Assign labels to all segments (fallback to Speaker 1)
            for seg in sentence_segments:
                key = (seg['start'], seg['end'])
                speaker_labels.append(speaker_map.get(key, "Speaker 1"))

        # Group consecutive same-speaker segments (no merging across different speakers)
        grouped = []
        if speaker_labels:
            curr = {
                'speaker': speaker_labels[0],
                'start': sentence_segments[0]['start'],
                'end': sentence_segments[0]['end'],
                'text': sentence_segments[0]['text']
            }
            for i in range(1, len(sentence_segments)):
                seg = sentence_segments[i]
                spk = speaker_labels[i]
                if spk == curr['speaker']:
                    curr['end'] = seg['end']
                    curr['text'] += ' ' + seg['text']
                else:
                    grouped.append(curr)
                    curr = {
                        'speaker': spk,
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text']
                    }
            grouped.append(curr)

        # Write CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['speaker', 'timestamp', 'End Time', 'words'])
            writer.writeheader()
            for seg in grouped:
                writer.writerow({
                    'speaker': seg['speaker'],
                    'timestamp': f"[{format_timestamp(seg['start'])}]",
                    'End Time': f"[{format_timestamp(seg['end'])}]",
                    'words': seg['text'].strip()
                })

        logger.info(f"✅ Output written to: {output_csv_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}")
        return False
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

# === MAIN ===

def main():
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.mp4', '.mov', '.avi', '.mkv', '.webm']
    input_files = [f for ext in extensions for f in Path(INPUT_DIR).glob(f"*{ext}")]

    if not input_files:
        logger.error(f"No files found in {INPUT_DIR}")
        return

    logger.info(f"Processing {len(input_files)} files...")

    successes = 0
    for f in input_files:
        output_path = Path(OUTPUT_DIR) / f"{f.stem}.csv"
        if process_file(f, output_path):
            successes += 1

    logger.info(f"✅ Completed: {successes}/{len(input_files)} files")

if __name__ == "__main__":
    main()