# STANDARD LIBRARY
import argparse
import datetime
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

# MONAI IMPORTS
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR

# CUSTOM IMPORTS
from scripts.utils.dataloader import get_loader
from scripts.utils.losses import IgnoreAwareDiceCELoss, RegionDiceBCELoss
from scripts.utils.trainer_CA_LR import trainer_CA_LR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_experiment_folder(log_root: str, checkpoint_root: str):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

    experiment_dir = os.path.join(log_root, timestamp)
    checkpoint_dir = os.path.join(checkpoint_root, timestamp)

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir


def configure_logging(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("swinunetr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def log_config(logger, name, config_dict):
    logger.info("======== %s ========", name)
    formatted = yaml.dump(config_dict, sort_keys=False, default_flow_style=False)
    for line in formatted.splitlines():
        logger.info(line)
    logger.info("====================")


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def build_model(model_config: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = model_config.get("model", {})
    feature_size = int(model_cfg.get("feature_size", 48))

    model = SwinUNETR(
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 4)),
        feature_size=feature_size,
        use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        dropout_path_rate=float(model_cfg.get("dropout_path_rate", 0.0)),
        use_v2=bool(model_cfg.get("use_v2", False)),
    ).to(device)

    return model


def run_experiment(
    train_config_path: str,
    model_config_path: str,
    checkpoint_path: str | None = None,
    train: bool = True,
    fold: int | None = None,
    epochs: int | None = None,
    val_every: int | None = None,
):
    if not train:
        raise ValueError("`train` must be True (this pipeline is training-only).")

    train_config = load_yaml(resolve_path(train_config_path))
    model_config = load_yaml(resolve_path(model_config_path))

    data_cfg = train_config.get("data", {})
    training_cfg = train_config.get("training", {})
    optimizer_cfg = train_config.get("optimizer", {})
    scheduler_cfg = train_config.get("scheduler", {})
    loss_cfg = train_config.get("loss", {})
    logging_cfg = train_config.get("logging", {})
    checkpoint_cfg = train_config.get("checkpoint", {})
    performance_cfg = train_config.get("performance", {})

    roi = tuple(data_cfg.get("roi", [128, 128, 128]))
    num_workers = data_cfg.get("num_workers", 4)
    data_dir = resolve_path(data_cfg.get("root_dir", "./data"))
    json_list = resolve_path(data_cfg.get("json_list", "./data/datalist.example.json"))
    pseudo_image_root = resolve_path(data_cfg.get("pseudo_image_root")) if data_cfg.get("pseudo_image_root") else None
    pseudo_label_root = resolve_path(data_cfg.get("pseudo_label_root")) if data_cfg.get("pseudo_label_root") else None
    batch_size = int(data_cfg.get("batch_size", 8))
    sw_batch_size = int(data_cfg.get("sw_batch_size", batch_size))
    infer_overlap = float(data_cfg.get("infer_overlap", 0.5))

    fold = int(data_cfg.get("fold", 0) if fold is None else fold)
    max_epochs = int(training_cfg.get("epochs", 300) if epochs is None else epochs)
    val_every = int(training_cfg.get("val_every", 10) if val_every is None else val_every)
    seed = int(training_cfg.get("seed", 42))

    learning_rate = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))

    log_root = resolve_path(logging_cfg.get("log_dir", "./outputs/runs"))
    checkpoint_root = resolve_path(checkpoint_cfg.get("save_dir", "./checkpoints"))
    experiment_dir, ckpt_dir = setup_experiment_folder(log_root, checkpoint_root)

    logger = configure_logging(os.path.join(experiment_dir, "run.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Throughput backend flags (no effect on what the model learns) ---
    _AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    amp_enabled = bool(performance_cfg.get("amp", False)) and device.type == "cuda"
    amp_dtype = _AMP_DTYPES.get(str(performance_cfg.get("amp_dtype", "bfloat16")).lower(), torch.bfloat16)
    if amp_dtype == torch.float32:
        amp_enabled = False

    if device.type == "cuda":
        if bool(performance_cfg.get("cudnn_benchmark", True)):
            torch.backends.cudnn.benchmark = True
        if bool(performance_cfg.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # GradScaler is only needed for fp16; bf16 has the dynamic range to skip it.
    scaler = torch.amp.GradScaler("cuda") if (amp_enabled and amp_dtype == torch.float16) else None

    logger.info("Starting SwinUNETR run")
    logger.info("AMP enabled: %s (dtype=%s, grad_scaler=%s)", amp_enabled, amp_dtype, scaler is not None)
    logger.info("Repository root: %s", REPO_ROOT)
    logger.info("Using device: %s", device)
    logger.info("Fold: %s", fold)
    logger.info("Data directory: %s", data_dir)
    logger.info("Datalist JSON: %s", json_list)
    logger.info("Pseudo image root: %s", pseudo_image_root)
    logger.info("Pseudo label root: %s", pseudo_label_root)
    log_config(logger, "TRAIN CONFIG", train_config)
    log_config(logger, "MODEL CONFIG", model_config)
    save_config(train_config, os.path.join(experiment_dir, "train_config.yaml"))
    save_config(model_config, os.path.join(experiment_dir, "model_config.yaml"))

    set_seed(seed)

    checkpoint_path = checkpoint_path or checkpoint_cfg.get("resume")
    checkpoint_path = resolve_path(checkpoint_path)

    model = build_model(model_config, device)

    loaded_checkpoint = None
    if checkpoint_path is not None:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = loaded_checkpoint["state_dict"] if "state_dict" in loaded_checkpoint else loaded_checkpoint
        model.load_state_dict(state_dict)
        logger.info("Checkpoint loaded successfully")

    num_classes = int(model_config.get("model", {}).get("out_channels", 4))
    if num_classes == 3:
        scheme = "regions"
        class_names = ["et", "tc", "wt"]
        score_channels = [0, 1, 2]
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = AsDiscrete(threshold=0.5)
    elif num_classes == 4:
        scheme = "multiclass"
        class_names = ["bg", "ncr", "ed", "et"]
        score_channels = [1, 2, 3]
        post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        post_label = AsDiscrete(to_onehot=num_classes)
    else:
        raise ValueError(
            f"model.out_channels must be 3 (overlapping ET/TC/WT regions, sigmoid) or "
            f"4 (mutually-exclusive bg/NCR/ED/ET, softmax), got {num_classes}"
        )
    logger.info("Output scheme: %s (%d channels: %s)", scheme, num_classes,
                ", ".join(class_names))
    min_dims = [32, 32, 32]

    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    optimizer_name = optimizer_cfg.get("type", "AdamW")
    if optimizer_name != "AdamW":
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_name = scheduler_cfg.get("type", "CosineAnnealingWarmRestarts")
    if scheduler_name != "CosineAnnealingWarmRestarts":
        raise ValueError(f"Unsupported scheduler type: {scheduler_name}")
    t_mult_raw = scheduler_cfg.get("t_mult", 1)
    if not isinstance(t_mult_raw, int):
        raise ValueError(f"scheduler.t_mult must be an integer >= 1, got {t_mult_raw!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_cfg.get("t_0", 100)),
        T_mult=t_mult_raw,
        eta_min=float(scheduler_cfg.get("eta_min", 1e-9)),
    )

    if scheme == "regions":
        dice_loss = RegionDiceBCELoss(
            smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
            smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
            squared_pred=bool(loss_cfg.get("squared_pred", True)),
            lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
            lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
        )
    else:
        dice_loss = IgnoreAwareDiceCELoss(
            include_background=bool(loss_cfg.get("include_background", True)),
            smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
            smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
            squared_pred=bool(loss_cfg.get("squared_pred", True)),
            lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
            lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
            ignore_index=int(loss_cfg.get("ignore_index", 4)),
        )

    # Down-weight the (noisier) pseudo-label signal relative to ground truth:
    # Loss = Loss_gt + pseudo_lambda * Loss_pseudo.
    pseudo_lambda = float(loss_cfg.get("pseudo_lambda", 0.5))

    start_epoch = 0
    resume_best_acc = 0.0
    if loaded_checkpoint is not None:
        if "optimizer_state_dict" in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in loaded_checkpoint:
            scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
        start_epoch = loaded_checkpoint.get("epoch", -1) + 1
        resume_best_acc = float(loaded_checkpoint.get("best_acc", 0.0))
        logger.info("Resuming training from epoch %d (best acc so far: %.4f)", start_epoch, resume_best_acc)

    periodic_save_every = int(checkpoint_cfg.get("periodic_save_every", 100))
    checkpoint_path_out = os.path.join(ckpt_dir, f"model_best_fold_{fold}.pth")
    val_acc_max = trainer_CA_LR(
        fold=fold,
        model=model,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        get_loader=get_loader,
        data_dir=data_dir,
        batch_size=batch_size,
        json_list=json_list,
        roi=roi,
        max_epochs=max_epochs,
        val_every=val_every,
        num_workers=num_workers,
        min_dims=min_dims,
        save_checkpoint_path=checkpoint_path_out,
        checkpoint_dir=ckpt_dir,
        periodic_save_every=periodic_save_every,
        logger=logger,
        device=device,
        val_acc_max=resume_best_acc,
        pseudo_image_root=pseudo_image_root,
        pseudo_label_root=pseudo_label_root,
        pseudo_lambda=pseudo_lambda,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        scaler=scaler,
        scheme=scheme,
        class_names=class_names,
        score_channels=score_channels,
    )
    logger.info("Training completed, best average dice: %.4f", val_acc_max)


def cli(default_mode: str | None = None):
    parser = argparse.ArgumentParser(description="Run training for SwinUNETR.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--fold", type=int, default=None, help="Fold override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--val-every", type=int, default=None, help="Validation frequency override.")
    parser.add_argument("--train", action="store_true", help="Run training.")

    args = parser.parse_args()

    run_experiment(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        train=args.train or default_mode == "train",
        fold=args.fold,
        epochs=args.epochs,
        val_every=args.val_every,
    )


def main(
    fold=0,
    max_epochs=300,
    val_every=20,
    model_path_file=None,
    model_train_flag=True,
):
    run_experiment(
        train_config_path="configs/train.yaml",
        model_config_path="configs/model.yaml",
        checkpoint_path=model_path_file,
        train=model_train_flag,
        fold=fold,
        epochs=max_epochs,
        val_every=val_every,
    )


if __name__ == "__main__":
    cli()
