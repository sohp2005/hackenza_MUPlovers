import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import librosa
import os

CHUNK_INDEX = Path("data/chunk_index.csv")
PROC_DIR = Path("data/processed")
VAD_PATH = Path("data/chunk_vad.csv")
OUT_DIR = Path("cache/features/prosody")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000

def pad_to_len(x, target_len):
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)), mode="constant")

def safe(x):
    return float(x) if np.isfinite(x) else 0.0

def prosody_features(chunk, speech_ratio):

    # Energy
    rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
    rms_mean = safe(np.mean(rms))
    rms_std = safe(np.std(rms))

    # Pitch (pyin)
    pitch_valid = 0.0
    f0_mean = f0_std = f0_slope = f0_range = 0.0
    try:
        f0, _, _ = librosa.pyin(
            chunk,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SR,
            frame_length=2048,
            hop_length=512
        )
        if f0 is not None:
            f0_voiced = f0[~np.isnan(f0)]
            if len(f0_voiced) >= 5:
                pitch_valid = 1.0
                f0_mean = safe(np.mean(f0_voiced))
                f0_std = safe(np.std(f0_voiced))
                f0_range = safe(np.percentile(f0_voiced,95) - np.percentile(f0_voiced,5))
                x = np.arange(len(f0_voiced))
                f0_slope = safe(np.polyfit(x, f0_voiced, 1)[0])
    except:
        pass

    # Spectral Flux
    S = np.abs(librosa.stft(chunk, n_fft=1024, hop_length=512)) + 1e-8
    S_norm = S / np.sum(S, axis=0, keepdims=True)
    flux = np.sqrt(np.sum((S_norm[:,1:] - S_norm[:,:-1])**2, axis=0))
    spectral_flux = safe(np.mean(flux)) if flux.size else 0.0

    # Speaking rate proxy
    onset_env = librosa.onset.onset_strength(y=chunk, sr=SR, hop_length=512)
    peaks = librosa.util.peak_pick(
    onset_env,
    pre_max=3,
    post_max=3,
    pre_avg=3,
    post_avg=3,
    delta=0.2,
    wait=3
)
    speaking_rate = safe(len(peaks)/3.0)

    return np.array([
        safe(speech_ratio),
        rms_mean,
        rms_std,
        pitch_valid,
        f0_mean,
        f0_std,
        f0_slope,
        f0_range,
        speaking_rate,
        spectral_flux
    ], dtype=np.float32)

ci = pd.read_csv(CHUNK_INDEX).sort_values(["file_id","chunk_id"])
vad = pd.read_csv(VAD_PATH)

m = ci.merge(vad[["file_id","chunk_id","speech_ratio"]],
             on=["file_id","chunk_id"],
             how="left")

for file_id, g in tqdm(m.groupby("file_id"),
                       total=m["file_id"].nunique(),
                       desc="Prosody"):

    out = OUT_DIR / f"{file_id}.npy"
    if out.exists():
        continue

    audio, sr = sf.read(PROC_DIR / f"{file_id}.wav")
    if audio.ndim>1:
        audio = audio.mean(axis=1)

    g = g.sort_values("chunk_id")
    T = len(g)
    feats = np.zeros((T,10),dtype=np.float32)

    for i,row in enumerate(g.itertuples(index=False)):
        start,end = int(row.start_sample),int(row.end_sample)
        chunk = audio[start:min(end,len(audio))]
        chunk = pad_to_len(chunk,48000).astype(np.float32)
        feats[i] = prosody_features(chunk,row.speech_ratio)

    np.save(out,feats)

print("✅ Prosody saved")