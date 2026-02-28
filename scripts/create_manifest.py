import pandas as pd
from pathlib import Path

# ---------- PATHS ----------
SRC_PATH = Path("data/source_training.csv")
OUT_PATH = Path("data/train_manifest.csv")

# ---------- LOAD ----------
df = pd.read_csv(SRC_PATH)

# ---------- CLEAN ----------
df["nativity_status"] = df["nativity_status"].str.strip()

# ---------- MAP LABEL ----------
label_map = {
    "Native": 1,
    "Non-Native": 0
}

df["label"] = df["nativity_status"].map(label_map)

# check for unmapped values
if df["label"].isnull().any():
    print("❌ Some labels could not be mapped:")
    print(df[df["label"].isnull()]["nativity_status"].unique())
    raise ValueError("Fix label values in source_training.csv")

# ---------- ADD FILE ID ----------
df["file_id"] = df["dp_id"].astype(str)

# ---------- ADD PATHS ----------
df["raw_path"] = df["file_id"].apply(
    lambda x: f"data/raw/{x}.wav"
)

df["processed_path"] = df["file_id"].apply(
    lambda x: f"data/processed/{x}.wav"
)

# ---------- SELECT COLUMNS ----------
manifest = df[[
    "file_id",
    "audio_url",
    "nativity_status",
    "label",
    "language",
    "raw_path",
    "processed_path"
]]

# ---------- SAVE ----------
manifest.to_csv(OUT_PATH, index=False)

print("✅ train_manifest.csv created!")
print(f"Total samples: {len(manifest)}")