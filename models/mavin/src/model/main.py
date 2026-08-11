# STANDARD LIBRARY
import argparse
import datetime
import logging
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

# MONAI IMPORTS
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

# CUSTOM IMPORTS
from scripts.utils.dataloader import get_loader
from scripts.utils.tester import tester
from scripts.utils.trainer import trainer

from .mambaVisionUNet import MambaVisionUNet as Mavin


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    logger = logging.getLogger("mavin")
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


def _extract_model_depth(model_config: dict) -> int:
    depths = model_config.get("model", {}).get("encoder", {}).get("depths", 1)
    if isinstance(depths, list):
        return int(depths[0])
    return int(depths)


def build_model(model_config: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = model_config.get("model", {})
    data_cfg = model_config.get("data", {})

    img_size = data_cfg.get("roi", [128, 128, 128])
    feature_size = int(model_cfg.get("feature_size", 48))
    depth = _extract_model_depth(model_config)
    num_heads = model_cfg.get("num_heads", 16)
    d_state = model_cfg.get("d_state", 16)

    model = Mavin(
        img_size=tuple(img_size),
        in_channels=int(model_cfg.get("in_channels", 4)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        feature_size=feature_size,
        depths=depth,
        num_heads=num_heads,
        d_state=d_state,
        use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
    ).to(device)

    return model


def run_experiment(
    train_config_path: str,
    model_config_path: str,
    checkpoint_path: str | None = None,
    train: bool = False,
    test: bool = False,
    fold: int | None = None,
    epochs: int | None = None,
    val_every: int | None = None,
):
    if not train and not test:
        raise ValueError("At least one of `train` or `test` must be True.")

    train_config = load_yaml(resolve_path(train_config_path))
    model_config = load_yaml(resolve_path(model_config_path))

    data_cfg = train_config.get("data", {})
    training_cfg = train_config.get("training", {})
    optimizer_cfg = train_config.get("optimizer", {})
    scheduler_cfg = train_config.get("scheduler", {})
    loss_cfg = train_config.get("loss", {})
    logging_cfg = train_config.get("logging", {})
    checkpoint_cfg = train_config.get("checkpoint", {})

    roi = tuple(data_cfg.get("roi", [128, 128, 128]))
    data_dir = resolve_path(data_cfg.get("root_dir", "./data"))
    json_list = resolve_path(data_cfg.get("json_list", "./data/datalist.example.json"))
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

    logger.info("Starting Mavin run")
    logger.info("Repository root: %s", REPO_ROOT)
    logger.info("Using device: %s", device)
    logger.info("Fold: %s", fold)
    logger.info("Data directory: %s", data_dir)
    logger.info("Datalist JSON: %s", json_list)

    set_seed(seed)

    model_config.setdefault("data", {})
    model_config["data"]["roi"] = list(roi)
    model = build_model(model_config, device)

    checkpoint_path = checkpoint_path or checkpoint_cfg.get("resume")
    checkpoint_path = resolve_path(checkpoint_path)

    if checkpoint_path is not None:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        logger.info("Checkpoint loaded successfully")

    optimizer_name = optimizer_cfg.get("type", "AdamW")
    if optimizer_name != "AdamW":
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler_name = scheduler_cfg.get("type", "CosineAnnealingWarmRestarts")
    if scheduler_name != "CosineAnnealingWarmRestarts":
        raise ValueError(f"Unsupported scheduler type: {scheduler_name}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(scheduler_cfg.get("t_0", 100)),
        T_mult=int(scheduler_cfg.get("t_mult", 1)),
        eta_min=float(scheduler_cfg.get("eta_min", 5e-9)),
    )

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    dice_loss = DiceLoss(
        to_onehot_y=False,
        sigmoid=True,
        smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
        smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
        squared_pred=bool(loss_cfg.get("squared_pred", True)),
    )
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    if train:
        checkpoint_path_out = os.path.join(ckpt_dir, f"model_best_fold_{fold}.pth")
        val_acc_max = trainer(
            fold=fold,
            model=model,
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            scheduler=scheduler,
            model_inferer=model_inferer,
            start_epoch=0,
            post_sigmoid=post_sigmoid,
            post_pred=post_pred,
            get_loader=get_loader,
            data_dir=data_dir,
            batch_size=batch_size,
            json_list=json_list,
            roi=roi,
            max_epochs=max_epochs,
            val_every=val_every,
            save_checkpoint_path=checkpoint_path_out,
            decollate_batch=decollate_batch,
            logger=logger,
            device=device,
        )
        logger.info("Training completed, best average dice: %.4f", val_acc_max)

    if test:
        if checkpoint_path is None:
            raise ValueError("Testing requires `--checkpoint` or `checkpoint.resume` in configs/train.yaml.")
        val_acc_max, hd_acc_max = tester(
            device=device,
            data_dir=data_dir,
            json_list=json_list,
            model=model,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_sigmoid=post_sigmoid,
            post_pred=post_pred,
            logger=logger,
            decollate_batch=decollate_batch,
        )
        logger.info("Testing completed, best average dice: %.4f", val_acc_max)
        logger.info("Testing completed, best average HD95: %.4f", hd_acc_max)


def cli(default_mode: str | None = None):
    parser = argparse.ArgumentParser(description="Run training or testing for Mavin.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for resume/testing.")
    parser.add_argument("--fold", type=int, default=None, help="Fold override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--val-every", type=int, default=None, help="Validation frequency override.")
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--test", action="store_true", help="Run testing.")

    args = parser.parse_args()

    train = args.train or default_mode == "train"
    test = args.test or default_mode == "test"

    run_experiment(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        train=train,
        test=test,
        fold=args.fold,
        epochs=args.epochs,
        val_every=args.val_every,
    )


def main(
    fold=0,
    max_epochs=300,
    val_every=10,
    model_path_file=None,
    model_train_flag=False,
    model_test_flag=False,
):
    run_experiment(
        train_config_path="configs/train.yaml",
        model_config_path="configs/model.yaml",
        checkpoint_path=model_path_file,
        train=model_train_flag,
        test=model_test_flag,
        fold=fold,
        epochs=max_epochs,
        val_every=val_every,
    )


if __name__ == "__main__":
    cli()
