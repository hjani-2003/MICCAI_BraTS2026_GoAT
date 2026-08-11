# Models

| Directory | Architecture | Role |
|---|---|---|
| `nnunet/` | nnU-Net v2, 3d_fullres, self-configuring CNN | The submitted segmenter. Best single model at 0.820 average Dice, and the baseline the post-processing pipeline is applied to. |
| `swinunetr_v2/` | SwinUNETR-v2, shifted-window transformer encoder-decoder with residual CNN connections | Second-best single model (0.801 supervised). Trained supervised and self-supervised. |
| `mavin/` | MaViN, Mamba state-space + Swin attention hybrid | Weakest of the three (0.719 supervised, 0.776 self-supervised); included in both fusions. Vendored from <https://github.com/hjani-2003/mavin>, its own paper's release, under its own MIT license. |
| `hrnet/` | `SmallTumorHRSeg`, 3D HRNet-style with a full-resolution branch throughout | Not a competitor. The donor for the empty-mask fallback in `../postprocessing/`. |

## Entry points

```bash
# nnU-Net: prepare -> preprocess -> splits -> train -> predict -> evaluate
python nnunet/src/main.py --steps prepare preprocess splits --help

# SwinUNETR-v2
python swinunetr_v2/scripts/train.py --help

# MaViN
python mavin/scripts/train.py --help

# HRNet
python hrnet/src/main.py --help
```

Every path in each `configs/` file is a placeholder — edit them before running anything.

## Output scheme (SwinUNETR-v2)

`swinunetr_v2/configs/model.yaml` `out_channels` selects the training objective, and the
dataloader, loss, and validation metric all follow it:

| `out_channels` | Labels | Loss | Selection metric |
|---|---|---|---|
| `3` | overlapping regions `[ET, TC, WT]` (`ConvertToBraTSRegionsd`) | per-region sigmoid BCE + soft Dice (`RegionDiceBCELoss`) | mean Dice over ET, TC, WT |
| `4` | mutually-exclusive `bg/NCR/ED/ET` (`ConvertToMultiClassd`) | softmax Dice + cross-entropy (`IgnoreAwareDiceCELoss`) | mean Dice over NCR, ED, ET |

Any other value is rejected at startup. Pseudo-labels are stored as single-channel class
maps with `ignore_index=4` and only work with `out_channels: 4`; combining them with `3`
raises rather than training on silently mis-shaped targets.

## What was and was not tuned

`nnunet/` uses nnU-Net's defaults, unmodified — its self-configuration is part of what is
being compared, so overriding it would have evaluated a different method.

`swinunetr_v2/` and `mavin/` share one configuration of framework and literature
defaults: 128³ patches, AdamW, 300 epochs, cosine annealing with warm restarts, gradient
clipping at norm 1.0, validation every 10 epochs by sliding-window inference at 0.7
overlap. **No architecture-specific hyperparameter search was run for either**, so part
of the gap between them and nnU-Net may reflect that untuned shared configuration rather
than an inherent architectural difference.

nnU-Net is also evaluated with the mirroring test-time augmentation its configuration
enables by default, while the other two use none. Mirroring is usually worth a small
positive amount, plausibly of the same order as the 0.009 margin over the best fusion, so
part of nnU-Net's lead may follow from that rather than architecture.

## Two traps

**The nnU-Net modality channel order.** The submitted checkpoint's own `dataset.json`
mislabels its channels: it declares `t1c, t1n, t2w, t2f`, but the model was trained on
`t1c, t1n, t2f, t2w` — the last two swapped. Pass the order explicitly at inference;
`../ensembling/ensemble_infer.py` takes `--nnunet-modality-order t1c t1n t2f t2w` and
refuses to guess. A swap here yields predictions that look plausible and score badly,
with nothing in the logs to explain why. HRNet is a third order again, `t1c, t1n, t2w,
t2f`.

**HRNet's training set is 24 cases.** Not the full dataset: the 24 labelled cases on
which the primary predictor produced no tumour at all. Those cases are in nnU-Net's own
training set, so they are failures on data it had already seen, and there is no held-out
split. Treat it as a fix for one specific failure mode, not as evidence of general
small-tumour detection.
