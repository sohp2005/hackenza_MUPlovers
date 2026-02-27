# Hackenza 2026 — Native vs Non-Native Speaker Classification (Audio-only)

## Repo layout
- `src/` core library code
- `scripts/` runnable entrypoints (preprocess, feature extraction, train, predict)
- `notebooks/` Colab notebooks (optional)
- `data/` (local placeholders only; real data should live in Google Drive or elsewhere)
- `cache/` embeddings/features cache (do not commit)
- `runs/` training outputs (do not commit)
- `submissions/` prediction CSV (do not commit by default)

## Expected workflow
1. Create manifests in Drive/local.
2. Preprocess audio to 16k mono.
3. Chunk + VAD stats.
4. Extract wav2vec embeddings + prosody/noise features (cache).
5. Train GRU+attention+APL.
6. Predict test → `predictions.csv`.
