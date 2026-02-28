import pandas as pd
from sklearn.model_selection import train_test_split

manifest = pd.read_csv("data/train_manifest.csv")

file_ids = manifest["file_id"].unique()
labels = manifest.set_index("file_id").loc[file_ids]["label"]

# First split: Train (70%) vs Temp (30%)
train_ids, temp_ids = train_test_split(
    file_ids,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Labels for temp
temp_labels = manifest.set_index("file_id").loc[temp_ids]["label"]

# Second split: Val (15%) vs Test (15%)
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

pd.DataFrame({"file_id": train_ids}).to_csv("data/train_ids.csv", index=False)
pd.DataFrame({"file_id": val_ids}).to_csv("data/val_ids.csv", index=False)
pd.DataFrame({"file_id": test_ids}).to_csv("data/test_ids.csv", index=False)

print("✅ Split created:")
print("Train:", len(train_ids))
print("Val:", len(val_ids))
print("Test:", len(test_ids))