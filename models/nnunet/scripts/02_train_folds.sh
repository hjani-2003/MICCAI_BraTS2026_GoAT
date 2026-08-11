#!/usr/bin/env bash
# Step 2: train nnUNet on all 5 folds (or a single fold via --fold N).
#
# Usage:
#   bash scripts/02_train_folds.sh              # trains folds 0-4 sequentially
#   bash scripts/02_train_folds.sh --fold 0     # trains fold 0 only
#   bash scripts/02_train_folds.sh --fold 0 --continue  # resume interrupted run
set -euo pipefail

CONFIG="configs/paths.yaml"
FOLD="all"
CONTINUE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold)     FOLD="$2"; shift 2 ;;
        --continue) CONTINUE_FLAG="--c"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

eval "$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print('export nnUNet_raw=\"'          + c['nnunet_raw']          + '\"')
print('export nnUNet_preprocessed=\"' + c['nnunet_preprocessed'] + '\"')
print('export nnUNet_results=\"'      + c['nnunet_results']      + '\"')
print('DATASET_ID=\"'                 + str(c['dataset_id'])     + '\"')
print('CONFIGURATION=\"'             + c['configuration']       + '\"')
print('TRAINER=\"'                   + c.get('trainer', 'nnUNetTrainer') + '\"')
")"

train_fold() {
    local fold="$1"
    echo "======================================================"
    echo " Training fold $fold"
    echo "======================================================"
    nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$fold" \
        -tr "$TRAINER" $CONTINUE_FLAG
}

if [[ "$FOLD" == "all" ]]; then
    for fold in 0 1 2 3 4; do
        train_fold "$fold"
    done
else
    train_fold "$FOLD"
fi

echo ""
echo "Training complete. Next: python src/main.py --steps predict --all-folds"
