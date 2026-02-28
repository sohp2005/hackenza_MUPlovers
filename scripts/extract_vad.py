import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import csv

CHUNK_INDEX = Path("data/chunk_index.csv")
PROC_DIR = Path("data/processed")
OUT_VAD = Path("data/chunk_vad.csv")

SR = 16000
FRAME_MS = 30
FRAME_LEN = int(SR * FRAME_MS / 1000)

def pad_to_len(x, target_len):
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)), mode="constant")

def vad_stats(chunk):
    """
    chunk: float32 mono audio, length 48000
    returns: speech_ratio, pause_ratio, max_pause_len_sec
    """

    frames = len(chunk) // FRAME_LEN
    if frames == 0:
        return 0.0, 1.0, 3.0

    speech_flags = []

    for i in range(frames):
        frame = chunk[i*FRAME_LEN:(i+1)*FRAME_LEN]

        # RMS energy
        energy = np.sqrt(np.mean(frame**2))

        # threshold (tunable, but works well)
        speech_flags.append(1 if energy > 0.01 else 0)

    speech_flags = np.array(speech_flags)
    speech_ratio = float(np.mean(speech_flags))
    pause_ratio = 1.0 - speech_ratio

    # longest silence
    max_pause = 0
    cur = 0
    for f in speech_flags:
        if f == 0:
            cur += 1
            max_pause = max(max_pause, cur)
        else:
            cur = 0

    max_pause_len_sec = (max_pause * FRAME_MS) / 1000.0

    return speech_ratio, pause_ratio, max_pause_len_sec

df = pd.read_csv(CHUNK_INDEX)
df = df.sort_values(["file_id", "chunk_id"])

with open(OUT_VAD, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_id", "chunk_id", "speech_ratio", "pause_ratio", "max_pause_len_sec"])

    for file_id, g in tqdm(df.groupby("file_id"), total=df["file_id"].nunique(), desc="VAD"):
        wav_path = PROC_DIR / f"{file_id}.wav"
        audio, sr = sf.read(wav_path)
        if sr != SR:
            raise ValueError(f"{wav_path} sr={sr}, expected {SR}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        for row in g.itertuples(index=False):
            start = int(row.start_sample)
            end = int(row.end_sample)

            chunk = audio[start:min(end, len(audio))]
            chunk = pad_to_len(chunk, 48000).astype(np.float32)

            sratio, pratio, maxpause = vad_stats(chunk)
            writer.writerow([file_id, int(row.chunk_id), sratio, pratio, maxpause])

print("✅ Wrote:", OUT_VAD)