import os
import gc
import time
import numpy as np
import torch

from .train_epoch import train_epoch
from .valid_epoch import val_epoch
from .save_checkpoint import save_checkpoint


def trainer(
    fold,
    model,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer,
    start_epoch,
    post_softmax,
    post_pred,
    post_label,
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
):
    train_loader, val_loader = get_loader(
        batch_size, data_dir, json_list, fold, roi,
        min_dims=min_dims, num_workers=num_workers,
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
        )

        logger.info(f"Final training {epoch + 1}/{max_epochs}  Loss {train_loss:.4f}  Time {time.time() - epoch_time:.2f}s")

        for param_group in optimizer.param_groups:
            logger.info(f"LR: {param_group['lr']:.2e}")

        if (epoch + 1) % val_every == 0 or epoch == 0 or epoch + 1 == max_epochs:

            epoch_time = time.time()

            val_acc = val_epoch(
                device=device,
                max_epochs=max_epochs,
                model=model,
                loader=val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_softmax=post_softmax,
                post_pred=post_pred,
                post_label=post_label,
                logger=logger,
            )

            dice_bg, dice_ncr, dice_ed, dice_et = val_acc
            val_avg_acc = np.mean(val_acc[1:])  # exclude background from the headline average

            logger.info(
                f"Val {epoch}/{max_epochs - 1} | "
                f"BG {dice_bg:.4f}  NCR {dice_ncr:.4f}  ED {dice_ed:.4f}  ET {dice_et:.4f} | "
                f"Avg(fg) {val_avg_acc:.4f} | Time {time.time() - epoch_time:.2f}s"
            )

            if val_avg_acc > val_acc_max:
                logger.info(f"New best ({val_acc_max:.6f} → {val_avg_acc:.6f})")
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    logger=logger, epoch=epoch, path=save_checkpoint_path, best_acc=val_acc_max,
                )

        if periodic_save_every > 0 and (epoch + 1) % periodic_save_every == 0:
            periodic_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}_fold_{fold}.pth")
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                logger=logger, epoch=epoch, path=periodic_path, best_acc=val_acc_max,
            )

        scheduler.step()

    logger.info(f"Training finished. Best avg dice: {val_acc_max:.4f}")
    return val_acc_max
