# Hackenza 2026 — Native vs Non-Native Speaker Classification (Audio-only)

**Team:** MUPlovers  
**Members:** Soham Pujari (Lead), Nirek Agarwal, Harshal Lahoti, Aarush Goyal  
**Problem Setter:** Renan Partners Private Limited

---

## Overview

An end-to-end automated machine learning pipeline that classifies ~2-minute audio recordings as **native** or **non-native** speaker — using only the raw audio signal. No transcription, no manual labeling.

Each prediction comes with a **confidence score** (calibrated probability).

---

## Pipeline Architecture

```
Raw Audio (2 min)
        │
        ▼
┌─────────────────────────────┐
│   1. PREPROCESSING (S)      │
│   • Mono + resample 16kHz   │
│   • Amplitude normalization │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   2. CHUNKING + VAD (S)     │
│   • 3s windows, 1.5s hop    │
│   • ~70–80 chunks/file      │
│   • Speech/pause ratio      │
└────────────┬────────────────┘
             │
     ┌───────┼────────────────┐
     ▼       ▼                ▼
┌─────────┐ ┌──────────────┐ ┌──────────────┐
│WavLM-   │ │   PROSODY    │ │    NOISE     │
│Large    │ │   (S)        │ │    (A)       │
│Embed.   │ │  F0, energy  │ │  SNR, flux   │
│[T,1024] │ │  rhythm      │ │  centroid    │
│(N)      │ │  [T, 10]     │ │  [T, 5]      │
└────┬────┘ └──────┬───────┘ └──────┬───────┘
     │             │                │
     └─────────────┴────────────────┘
                   │  Concatenate
                   ▼
        ┌─────────────────────┐
        │  FEATURE ASSEMBLY   │
        │  Normalize [T,1039] │
        │       (A)           │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   MODEL (H)         │
        │  Projection Layer   │
        │  2-layer BiGRU      │
        │  Multi-head Attn    │
        │  BatchNorm          │
        │  MLP Classifier     │
        └──────────┬──────────┘
                   │
                   ▼
        Native / Non-Native + Confidence Score
```

---

## Feature Breakdown

| Feature Group | Dim | What it captures |
|---|---|---|
| WavLM-Large embeddings (weighted layers) | 1024 | Deep acoustic + phonetic patterns |
| Prosody | 10 | F0 mean/std/slope/range, energy, speaking rate, rhythm, spectral flux |
| Noise | 5 | SNR proxy, spectral entropy, flatness, centroid, bandwidth |
| **Total** | **1039** | |

---

## Model Architecture

```
Input [T, 1039]
    → Linear(1039, 256) + LayerNorm + ReLU        # Projection
    → BiGRU(256, 256, layers=2, bidirectional)    # Temporal modeling
    → MultiheadAttention(512, heads=4)            # Accent pattern focus
    → Mean Pool → BatchNorm(512)                  # Aggregation
    → Linear(512, 256) → ReLU → Dropout(0.3)     # Classifier
    → Linear(256, 1) → Sigmoid                   # Output
```

**Training details:**
- Loss: `BCEWithLogitsLoss` with class-weighted `pos_weight`
- Optimizer: Adam, lr=1e-4
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Epochs: 50
- Batch size: 8

---

## Datasets

Three datasets were used for training:

| Dataset | Source | Files |
|---|---|---|
| Main | Provided training data | ~160 files |
| External | Additional collected data | external/ |
| External2 | Additional collected data | external2/ |

All datasets go through the same preprocessing pipeline. Test set predictions are generated from `data/test_ids.csv` and results in 'results.csv' for the same. 
Please note all final versions of the code used are under notebooks_final directory.

---

## Repo Structure

```
hackenza_MUPlovers/
│
├── data/                          # Data manifests and metadata (no raw audio)
│   ├── chunk_index.csv            # Chunk boundaries for main dataset
│   ├── chunk_vad.csv              # VAD stats per chunk
│   ├── train_manifest.csv         # Full training manifest
│   ├── train_ids.csv              # Train split IDs
│   ├── val_ids.csv                # Val split IDs
│   ├── test_ids.csv               # Test file IDs (unlabeled)
│   ├── source_training.csv        # Raw source labels
│   └── splits/                    # Train/val split files
│
├── notebooks/                     # Iterative experiment notebooks
│   ├── HackN_Phase1FINAL.ipynb    # N: WavLM-Large weighted embeddings
│   ├── Harshal_Phase1FINAL.ipynb  # H: GRU + Attention training
│   ├── Feature_Assembly_1039.ipynb# A: Noise + assembly + normalization
│   ├── attempt1.ipynb             # Ablation: mean pooling baseline
│   ├── gru_only_model.ipynb       # Ablation: GRU only
│   ├── attention_pooling_only.ipynb# Ablation: attention only
│   └── ...
│
├── notebooks_final/               # Clean final versions for submission
│   ├── WavLM_Large_Enhanced.ipynb # Final embedding extraction
│   ├── Feature_Assembly_1039.ipynb# Final feature assembly
│   ├── GRUATTENTION_1039.ipynb    # Final model training
│   └── WavLM_Base_783.ipynb       # Baseline (783-dim) for ablation
│
├── scripts/                       # Standalone runnable Python scripts
│   ├── create_manifest.py         # Step 1: build manifest from source CSV
│   ├── download_preprocess_all.py # Step 2: download + preprocess audio
│   ├── make_chunk_index.py        # Step 3: build chunk index
│   ├── extract_vad.py             # Step 4: VAD stats per chunk
│   ├── extract_prosody.py         # Step 5: prosody features
│   ├── create_train_val_split.py  # Step 6: train/val split
│   └── smoke_download_preprocess.py # Quick smoke test
│
├── src/                           # Core reusable library code
│
├── runs/                          # Saved model checkpoints (not committed)
│   └── best_model.pt
│
├── submissions/                   # Final output (not committed by default)
│   └── predictions.csv
│
├── external/                      # External dataset 1 (Drive only)
│   ├── processed/                 # Preprocessed wavs
│   ├── cache/
│   │   ├── embeddings_1039/       # WavLM-Large embeddings [T,1024]
│   │   ├── features/prosody/      # Prosody features [T,10]
│   │   ├── noise/                 # Noise features [T,5]
│   │   ├── features_1039/         # Assembled [T,1039]
│   │   └── features_normalized_1039/ # Final normalized features
│   └── train_manifest_processed.csv
│
├── external2/                     # External dataset 2 (Drive only)
│   ├── processed/
│   ├── cache/
│   │   ├── embeddings_1039/
│   │   ├── features/prosody/
│   │   ├── noise/
│   │   ├── features_1039/
│   │   └── features_normalized_1039/
│   └── train_manifest_processed.csv
│
├── cache/                         # Main dataset cache (Drive only, not committed)
│   ├── embeddings_1039/           # WavLM-Large embeddings [T,1024]
│   ├── features/prosody/          # Prosody [T,10]
│   ├── noise/                     # Noise [T,5]
│   ├── features_1039/             # Assembled [T,1039]
│   ├── features_normalized_1039/  # Final normalized [T,1039]
│   └── scaler_1039.pkl            # Fitted StandardScaler
│
├── .gitignore
|-- results.csv
└── README.md
```

---

## How to Reproduce — Step by Step

### Prerequisites

```bash
pip install torch torchaudio transformers librosa soundfile scikit-learn tqdm pandas numpy
```

All heavy computation runs on Google Colab (T4 GPU). Data lives on Google Drive at:
```
/content/drive/MyDrive/Hackenza_MUPlovers/
```

---

### Step 1 — Preprocess Audio (S)

```bash
python scripts/create_manifest.py         # build manifest from source CSV
python scripts/download_preprocess_all.py # resample to 16kHz mono, normalize
python scripts/make_chunk_index.py        # 3s chunks, 1.5s hop
python scripts/extract_vad.py            # speech/pause ratio per chunk
python scripts/extract_prosody.py        # prosody features [T, 10]
python scripts/create_train_val_split.py # 80/20 stratified split
```

Repeat for `external/` and `external2/` using the corresponding `external_0X_*.py` scripts.

---

### Step 2 — Extract WavLM-Large Embeddings (N)

Run `notebooks_final/WavLM_Large_Enhanced.ipynb` on Colab (GPU required).

- Model: `microsoft/wavlm-large`
- Uses **weighted combination of all 25 hidden layers** (learnable weights)
- Output: `cache/embeddings_1039/{file_id}.npy` → shape `[T, 1024]`

Change paths at the top of the notebook for each dataset:
```python
PROCESSED_AUDIO_PATH = ".../<dataset>/processed/"
CHUNK_INDEX_PATH     = ".../<dataset>/chunk_index.csv"
EMBED_SAVE_PATH      = ".../<dataset>/cache/embeddings_1039/"
```

---

### Step 3 — Feature Assembly + Normalization (A)

Run `notebooks_final/Feature_Assembly_1039.ipynb` on Colab.

- Extracts noise features [T, 5] from audio chunks
- Concatenates: embeddings [T,1024] + prosody [T,10] + noise [T,5] = **[T, 1039]**
- Fits `StandardScaler` and normalizes
- Saves scaler to `cache/scaler_1039.pkl`

Change paths at the top for each dataset.

---

### Step 4 — Train Model (H)

Run `notebooks_final/GRUATTENTION_1039.ipynb` on Colab (GPU required).

Combines all 3 datasets via `ConcatDataset`:
```python
combined = ConcatDataset([train_main, train_ext, train_ext2])
```

Best model saved to `runs/best_model.pt`.

---

### Step 5 — Generate Predictions

Run the inference cells at the bottom of `GRUATTENTION_1039.ipynb`.

- Loads `runs/best_model.pt`
- Loads test file features using `data/test_ids.csv`
- Outputs `submissions/predictions.csv`

Format:
```
file_id, predicted_label, confidence
288, 1, 0.923
294, 0, 0.711
...
```

---

## Ablations

We ran the following ablations to justify design choices:

| Model | Val Accuracy |
|---|---|
| Mean pooling (baseline) | 0.8333 |
| GRU only | 0.9189 |
| GRU + Attention (783-dim, WavLM-Base) | 0.9167 |
| Full model (BiGRU + MultiheadAttn + BatchNorm) | 0.79 |

*(Fill in accuracy numbers after training)*

---

## Key Design Decisions

**Why chunk-based modeling?** 2-minute recordings are too long to process as a single unit. Chunking lets us capture local accent patterns while GRU captures long-range temporal structure.

**Why WavLM-Large over WavLM-Base?** Large model trained on more data gives richer acoustic representations (1024-dim vs 768-dim). Weighted layer combination lets the model learn which layers are most accent-informative rather than always using the last layer.

**Why BiGRU + Multi-head Attention?** Bidirectional GRU captures temporal context in both directions. Multi-head attention learns to focus on the most accent-revealing segments automatically.

**Why prosody + noise features?** Raw embeddings capture phonetics well but miss rhythm and recording quality. Explicit prosody features (F0, speaking rate, pause patterns) and noise features add complementary signal that embeddings alone miss.

---

## Constraints Satisfied

- ✅ Audio-only (no transcription)
- ✅ No manual labeling
- ✅ Fully automated pipeline (single command preprocess, single command predict)
- ✅ Outputs predicted label + confidence score per sample
- ✅ Fixed random seeds for reproducibility
