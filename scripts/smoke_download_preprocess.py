import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile

MANIFEST_PATH = Path("data/train_manifest.csv")
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_PATH)
sample = df.sample(5, random_state=42)

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for _, row in tqdm(sample.iterrows(), total=len(sample)):
    file_id = str(row["file_id"])
    url = row["audio_url"]

    raw_out = RAW_DIR / f"{file_id}.bin"          # store downloaded bytes (any format)
    proc_out = PROC_DIR / f"{file_id}.wav"        # final standardized wav

    try:
        # 1) download
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw_out.write_bytes(r.content)

        # 2) convert to 16k mono wav via ffmpeg
        # -vn: no video, -ac 1 mono, -ar 16000 sample rate
        run([
            "ffmpeg", "-y",
            "-i", str(raw_out),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(proc_out)
        ])

        # 3) keep raw bytes (optional). For smoke test, delete raw to save space:
        raw_out.unlink(missing_ok=True)

        print(f"✅ {file_id} -> {proc_out}")

    except Exception as e:
        print(f"❌ Failed {file_id}: {e}")
        # keep raw_out for debugging if it exists