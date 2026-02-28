import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROC_DIR = Path("data/processed")
OUT_PATH = Path("data/chunk_index.csv")

CHUNK_SEC = 3.0
HOP_SEC = 1.5
SR = 16000

rows = []

files = list(PROC_DIR.glob("*.wav"))

for f in tqdm(files):
    file_id = f.stem

    audio, sr = sf.read(f)
    total_samples = len(audio)
    duration = total_samples / SR

    chunk_len = int(CHUNK_SEC * SR)
    hop_len = int(HOP_SEC * SR)

    start = 0
    chunk_id = 0

    while start < total_samples:

        end = start + chunk_len
        is_padded = 0

        if end > total_samples:
            end = total_samples
            is_padded = 1

        rows.append([
            file_id,
            chunk_id,
            start / SR,
            end / SR,
            start,
            end,
            is_padded
        ])

        start += hop_len
        chunk_id += 1

df = pd.DataFrame(rows, columns=[
    "file_id",
    "chunk_id",
    "start_sec",
    "end_sec",
    "start_sample",
    "end_sample",
    "is_padded"
])

df.to_csv(OUT_PATH, index=False)

print("✅ chunk_index.csv created!")