#!/usr/bin/env bash
# Step 1: plan and preprocess the BraTS dataset.
# Run this once before training any fold.
#
# Usage: bash scripts/01_plan_and_preprocess.sh [config]
set -euo pipefail

CONFIG="${1:-configs/paths.yaml}"

# Export nnUNet environment variables from paths.yaml
eval "$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print('export nnUNet_raw=\"'          + c['nnunet_raw']          + '\"')
print('export nnUNet_preprocessed=\"' + c['nnunet_preprocessed'] + '\"')
print('export nnUNet_results=\"'      + c['nnunet_results']      + '\"')
print('DATASET_ID=\"'                 + str(c['dataset_id'])     + '\"')
")"

echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity -c 3d_fullres

echo ""
echo "Done. Next: generate_nnunet_splits.py, then run scripts/02_train_folds.sh"
