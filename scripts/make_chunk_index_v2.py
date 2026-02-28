import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROC_DIR = Path("data/processed")
OUT_PATH = Path("data/chunk_index.csv")  # overwrite the old one

CHUNK_SEC = 3.0
HOP_SEC = 1.5
TARGET_SR = 16000

chunk_len = int(CHUNK_SEC * TARGET_SR)   # 48000
hop_len = int(HOP_SEC * TARGET_SR)       # 24000

rows = []
files = sorted(PROC_DIR.glob("*.wav"))

for f in tqdm(files, desc="Building chunk_index"):
    file_id = f.stem

    info = sf.info(f)
    sr = info.samplerate
    total_samples = info.frames

    if sr != TARGET_SR:
        raise ValueError(f"{f} has sr={sr}, expected {TARGET_SR}. Re-run preprocess.")

    start = 0
    chunk_id = 0

    while start < total_samples:
        end = start + chunk_len
        is_padded = 1 if end > total_samples else 0

        rows.append([
            file_id,
            chunk_id,
            start / TARGET_SR,
            end / TARGET_SR,
            start,
            end,
            chunk_len,
            is_padded,
            total_samples
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
    "chunk_len_samples",
    "is_padded",
    "file_total_samples"
])

df.to_csv(OUT_PATH, index=False)

print("✅ chunk_index.csv (v2) created")
print("rows:", len(df))
print("unique files:", df["file_id"].nunique())
print("avg chunks/file:", df.groupby("file_id")["chunk_id"].max().add(1).mean())