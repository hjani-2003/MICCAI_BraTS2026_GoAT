"""Generate a labeled-only dataset JSON with fold assignments taken from
already_splitted.json.

already_splitted.json is a list of 5 dicts, each with "train" and "val" keys
containing case IDs like "BraTS-GoAT-XXXXX".

Pseudo-labels are NOT part of this JSON: scripts/utils/dataloader.py:get_loader
discovers them directly from a directory at train time (configs/train.yaml's
data.pseudo_label_dir + data.unlabeled_root), populated by
src/generate_pseudo_labels.py. The only step to pick up a fresh batch of
pseudo-labels is pointing pseudo_label_dir at that run's output dir -- no JSON
rebuild needed.
"""

import json

LABELED_JSON = "/path/to/folds.json"
SPLITS_JSON  = "/path/to/folds_5.json"
OUT_JSON     = "data/semi_supervised_dataset.json"

# --- Build case_id -> fold mapping from val sets ---
with open(SPLITS_JSON) as f:
    splits = json.load(f)

case_to_fold = {}
for fold_idx, split in enumerate(splits):
    for case_id in split["val"]:
        case_to_fold[case_id] = fold_idx

print(f"Fold assignment loaded: {len(case_to_fold)} labeled cases across {len(splits)} folds")

# --- Labeled entries with corrected fold numbers ---
with open(LABELED_JSON) as f:
    labeled_data = json.load(f)

training_entries = []
missing_from_splits = []

for entry in labeled_data["training"]:
    case_id = entry["label"].split("/")[0]
    if case_id not in case_to_fold:
        missing_from_splits.append(case_id)
        continue
    training_entries.append({
        "fold":  case_to_fold[case_id],
        "image": entry["image"],
        "label": entry["label"],
    })

if missing_from_splits:
    print(f"WARNING: {len(missing_from_splits)} labeled case(s) not found in splits and skipped:")
    for m in missing_from_splits:
        print(f"  {m}")

# --- Write output ---
out = {"training": training_entries}

with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved {OUT_JSON}")
print(f"  labeled : {len(training_entries)}")

# Sanity check: verify per-fold counts match already_splitted
from collections import Counter
fold_counts = Counter(e["fold"] for e in training_entries)
print("\nFold distribution (val sizes):")
for fold_idx, split in enumerate(splits):
    expected_val = len(split["val"])
    actual_val   = fold_counts.get(fold_idx, 0)
    status = "OK" if expected_val == actual_val else "MISMATCH"
    print(f"  Fold {fold_idx}: expected val={expected_val}, got={actual_val} [{status}]")
