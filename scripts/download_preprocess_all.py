import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import csv

MANIFEST_PATH = Path("data/train_manifest.csv")

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
LOG_DIR = Path("data/logs")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_PATH)

log_path = LOG_DIR / "download_preprocess_log.csv"

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

with open(log_path, "w", newline="") as logf:
    writer = csv.writer(logf)
    writer.writerow(["file_id", "status", "error"])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = str(row["file_id"])
        url = row["audio_url"]

        raw_out = RAW_DIR / f"{file_id}.bin"
        proc_out = PROC_DIR / f"{file_id}.wav"

        if proc_out.exists():
            writer.writerow([file_id, "skipped", "already exists"])
            continue

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            raw_out.write_bytes(r.content)

            run([
                "ffmpeg", "-y",
                "-i", str(raw_out),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                str(proc_out)
            ])

            raw_out.unlink(missing_ok=True)
            writer.writerow([file_id, "success", ""])

        except Exception as e:
            writer.writerow([file_id, "fail", str(e)])