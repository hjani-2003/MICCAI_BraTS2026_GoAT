# A Self-Configuring Model versus Fused Ensembles for Brain Tumor Segmentation on the BraTS-GoAT Benchmark

Code for our BraTS 2026 GoAT (Generalized and Automated Tumor Segmentation) Challenge
**Task 3** entry.

Harman Jani, Mehul S. Raval, Jayendra M. Bhalodiya — Ahmedabad University, India

Paper: [OpenReview](https://openreview.net/forum?id=xcByIqAUwJ)

## What this work found

We trained three architecturally distinct networks on all five GoAT sub-tasks (adult
glioma, BraTS-Africa glioma, meningioma, brain metastases, pediatric tumors) and expected
that fusing them would beat any one of them. It did not. Supervised nnU-Net alone scored
above both fusion strategies we tested, and stayed ahead when the identical
post-processing was applied to the best fusion. A paired per-subject test does not find
the remaining difference significant (p=0.40), so the defensible claim is that fusion
*did not help* here — not that the single model is reliably better.

A two-stage inference-time post-processing step then raised the single model from 0.820 to
**0.838** average Dice on the 451-subject online validation set, with no retraining. Part
of that gain is a scoring asymmetry in the official evaluator rather than better
segmentation; the paper separates the two.

Dice on the BraTS 2026 GoAT online validation set (451 subjects):

| Configuration | ET | TC | WT | Avg |
|---|---|---|---|---|
| MaViN, supervised | 0.676 | 0.723 | 0.759 | 0.719 |
| MaViN, self-supervised | 0.727 | 0.764 | 0.835 | 0.776 |
| SwinUNETR-v2, supervised | 0.759 | 0.789 | 0.855 | 0.801 |
| SwinUNETR-v2, self-supervised | 0.734 | 0.767 | 0.844 | 0.782 |
| **nnU-Net, supervised** | **0.770** | **0.816** | **0.874** | **0.820** |
| All three, equal average | 0.761 | 0.791 | 0.858 | 0.803 |
| All three, weighted average (grid search) | 0.765 | 0.804 | 0.865 | 0.811 |
| nnU-Net + ET size filter (T=75) | 0.797 | 0.816 | 0.874 | 0.829 |
| **nnU-Net + filter + fallback (submitted)** | **0.806** | **0.824** | **0.883** | **0.838** |
| Best fusion + the same two steps | 0.798 | 0.820 | 0.873 | 0.830 |

## Layout

```
models/
├── nnunet/          nnU-Net v2 3d_fullres — the submitted segmenter
├── swinunetr_v2/    SwinUNETR-v2, supervised and self-supervised
├── mavin/           MaViN (Mamba + Swin hybrid), vendored from its own release
└── hrnet/           SmallTumorHRSeg — the empty-mask fallback donor
ensembling/          fused inference + the out-of-fold weight grid search
postprocessing/      the two submitted inference-time rules
```

See [`models/README.md`](models/README.md) for entry points and the two channel-order and
training-scheme traps worth knowing before you run anything.

## The submitted pipeline

The supervised nnU-Net five-fold ensemble (`nnUNetv2_predict -f 0 1 2 3 4`,
sliding-window, thresholded at 0.5, with nnU-Net's own mirroring test-time augmentation),
followed by two inference-time rules:

1. **Empty-mask fallback.** On 13 of 451 subjects nnU-Net returns a completely empty
   volume while ground truth has a lesion in all three regions. Where the primary
   prediction has zero foreground voxels, substitute HRNet's.
2. **ET size filter.** Drop predicted enhancing-tumor connected components smaller than
   T=75 voxels, relabelling them to necrotic core (label 1) rather than background, so
   tumor core and whole tumor stay bit-identical and only ET can move.

Donor first, filter last — reversing them is self-defeating, because the filter would
empty ET on subjects whose components are all sub-threshold and the donor would
immediately re-fill it.

```bash
python postprocessing/run_postprocessing.py \
    --pred-dir  /path/to/nnunet_predictions \
    --donor-dir /path/to/hrnet_predictions \
    --out-dir   submission_final --min-et-voxels 75
```

Both triggers are computed from the model's own output and need no ground truth, so both
are legitimate at test time.

## Reading the post-processing gain honestly

**Most of the ET size filter's headline gain is an evaluator artifact.** The official
evaluator drops a subject from the ET mean when prediction and ground truth are both
ET-empty, but scores a non-empty prediction against empty ground truth as a hard DSC of 0.
Ground truth has no ET on 32 of the 451 subjects, so a few stray voxels earn a
subject-level zero. The filter moves 18 subjects from a scored zero to `undefined`, and
that denominator change is its entire positive contribution: among the 422 subjects still
scored on ET it *lowers* mean ET Dice by 0.005 and hurts more than it helps (103 worse, 67
better, p=0.0003). Its HD95 gain has the same origin.

**The fallback's gain rests on six subjects** — it fires on 13, substitutes on 8, and
changes the score of only 6. All 6 improve (p=0.031), but that is a thin base.

**`T=75` and the donor choice were fitted on the same 451 subjects we report on**, across
36 leaderboard submissions, rather than fixed in advance. Every `T` in [25, 200] is
positive, so the choice sits on a plateau rather than a spike, but that is a robustness
observation, not evidence of generalisation.

**No per-cohort breakdown was available**, so every claim concerns the pooled aggregate.
Despite the benchmark's framing, this work makes no claim about cross-cohort
generalisation.

## Data and weights

The BraTS-GoAT dataset (1,351 labelled and 1,138 unlabelled cases across five tumour
types) is distributed by the challenge organizers via Synapse under their own data-use
agreement: <https://www.synapse.org/brats2026>. Nothing here redistributes imaging data,
labels, or evaluator score exports.

All models were trained on a shared five-fold split of the 1,351 labelled cases (≈270 per
fold, formed by shuffling the pooled case list from all five sub-tasks). Trained
checkpoints are not in this repository; request them from the corresponding author
(`harman.jani@ahduni.edu.in`).

Hardware: nnU-Net on an NVIDIA RTX 5090; SwinUNETR-v2, MaViN and HRNet on an NVIDIA
H100 NVL.

## Citation

Paper: [OpenReview](https://openreview.net/forum?id=xcByIqAUwJ)

```bibtex
@inproceedings{jani2026goat,
  title     = {A Self-Configuring Model versus Fused Ensembles for Brain Tumor
               Segmentation on the BraTS-GoAT Benchmark},
  author    = {Jani, Harman and Raval, Mehul S. and Bhalodiya, Jayendra M.},
  booktitle = {Brain Tumor Segmentation (BraTS) Challenge, MICCAI},
  year      = {2026}
}
```

MaViN has its own paper; if you use `models/mavin/`, cite that as well.

See [REFERENCES.md](REFERENCES.md) for the full reference list from the paper.

## License

MIT — see [LICENSE](LICENSE). `models/mavin/` carries its own upstream MIT license.

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and
[MONAI](https://github.com/Project-MONAI/MONAI).

Supported by the University Challenge Grant of Ahmedabad University (Grant Ref. No.
URBSEASI25A3), with computational resources from the Stepwell High Performance Computing
facility at Ahmedabad University.
