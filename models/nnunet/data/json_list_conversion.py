from pathlib import Path
import numpy as np
import os
import json


def get_paths(root_dir, append_gz=True):
    train_root_dir = Path(root_dir)
    samples = []

    for subject_folder in train_root_dir.iterdir():
        if not subject_folder.is_dir():
            continue

        subject_name = subject_folder.name
        rel_base = Path(subject_name)
        ext = ".nii.gz" if append_gz else ".nii"

        t1n = rel_base / f"{subject_name}-t1n{ext}"
        t1c = rel_base / f"{subject_name}-t1c{ext}"
        t2w = rel_base / f"{subject_name}-t2w{ext}"
        t2f = rel_base / f"{subject_name}-t2f{ext}"
        seg = rel_base / f"{subject_name}-seg{ext}"

        samples.append((str(t1n), str(t1c), str(t2w), str(t2f), str(seg)))

    return samples


all_samples = get_paths(
    "/path/to/BraTS-GoAT-TrainingData-With-GroundTruth",
    append_gz=True,
)

n_samples = len(all_samples)
indices = np.arange(n_samples)

np.random.seed(42)
np.random.shuffle(indices)

folds = np.array_split(indices, 6)

training_entries = []
for fold_id, fold_indices in enumerate(folds):
    for idx in fold_indices:
        t1n, t1c, t2w, t2f, seg = all_samples[idx]
        training_entries.append({
            "fold": fold_id,
            "image": [t1c, t1n, t2w, t2f],
            "label": seg,
        })

dataset_json = {"training": training_entries}

out_path = "dataset.json"
with open(out_path, "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"Saved dataset JSON with {len(training_entries)} training entries")
