import pandas as pd
from sklearn.model_selection import train_test_split

manifest = pd.read_csv("data/train_manifest.csv")

train_ids, val_ids = train_test_split(
    manifest["file_id"].unique(),
    test_size=0.2,
    random_state=42,
    stratify=manifest["label"]
)

pd.DataFrame({"file_id": train_ids}).to_csv("data/train_ids.csv", index=False)
pd.DataFrame({"file_id": val_ids}).to_csv("data/val_ids.csv", index=False)

print("✅ train/val split created")
print("train:", len(train_ids))
print("val:", len(val_ids))