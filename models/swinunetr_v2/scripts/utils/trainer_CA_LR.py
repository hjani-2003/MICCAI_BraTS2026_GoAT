import os
import time
import gc
import numpy as np
import torch

from .train_epoch import train_epoch
from .valid_epoch import val_epoch
from .save_checkpoint import save_checkpoint


def trainer_CA_LR(
    fold,
    model,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer,
    start_epoch,
    post_label,
    post_pred,
    get_loader,
    data_dir,
    batch_size,
    json_list,
    roi,
    max_epochs,
    val_every,
    num_workers,
    min_dims,
    save_checkpoint_path,
    checkpoint_dir,
    periodic_save_every,
    logger,
    device,
    val_acc_max=0.0,
    amp_enabled=False,
    amp_dtype=torch.bfloat16,
    scaler=None,
    pseudo_image_root=None,
    pseudo_label_root=None,
    pseudo_lambda=1.0,
    scheme="multiclass",
    class_names=None,
    score_channels=None,
):
    train_loader, val_loader = get_loader(
        batch_size,
        data_dir,
        json_list,
        fold,
        roi,
        min_dims=min_dims,
        num_workers=num_workers,
        pseudo_image_root=pseudo_image_root,
        pseudo_label_root=pseudo_label_root,
        scheme=scheme,
    )

    for epoch in range(start_epoch, max_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"{time.ctime()} - Epoch {epoch}")
        epoch_time = time.time()

        train_loss = train_epoch(
            device=device,
            max_epochs=max_epochs,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            loss_func=loss_func,
            logger=logger,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            pseudo_lambda=pseudo_lambda,
        )

        logger.info(f"Final training {epoch + 1}/{max_epochs}")
        logger.info(f"Loss {train_loss:.4f}")
        logger.info(f"Time {time.time() - epoch_time:.2f}s")

        for param_group in optimizer.param_groups:
            logger.info(f"Current learning rate: {param_group['lr']}")

        if (epoch + 1) % val_every == 0 or epoch == 0 or (epoch + 1 == max_epochs):
            epoch_time = time.time()

            val_acc = val_epoch(
                device=device,
                max_epochs=max_epochs,
                model=model,
                loader=val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
                logger=logger,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                class_names=class_names,
            )

            names = class_names or [f"c{i}" for i in range(len(val_acc))]
            scored = score_channels if score_channels is not None else range(len(val_acc))
            val_avg_acc = float(np.mean([val_acc[i] for i in scored]))

            logger.info(f"Final validation stats {epoch}/{max_epochs - 1}")
            for name, value in zip(names, val_acc):
                logger.info(f"dice_{name} {value:.4f}")
            scored_label = "/".join(names[i].upper() for i in scored)
            logger.info(f"Dice_Avg ({scored_label}) {val_avg_acc:.4f}")
            logger.info(f"Time {time.time() - epoch_time:.2f}s")

            if val_avg_acc > val_acc_max:
                logger.info(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    epoch=epoch,
                    path=save_checkpoint_path,
                    best_acc=val_acc_max,
                )

        if periodic_save_every > 0 and (epoch + 1) % periodic_save_every == 0:
            periodic_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}_fold_{fold}.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                epoch=epoch,
                path=periodic_path,
                best_acc=val_acc_max,
            )

        scheduler.step()

    logger.info(f"Training Finished! Best Accuracy: {val_acc_max:.6f}")
    return val_acc_max
