import gc
import os
import time

import numpy as np
import torch
from torch import nn

from .train_epoch import train_epoch
from .valid_epoch import valid_epoch


def trainer(
    fold,
    model,
    
    optimizer,
    
    loss_func,
    
    acc_func,
    
    scheduler,
    
    model_inferer,
    start_epoch,

    post_sigmoid,
    post_pred,

    get_loader,
    data_dir,
    batch_size,
    json_list,
    roi,
    max_epochs,
    val_every,

    save_checkpoint_path,
    decollate_batch,
    
    logger,
    device
):
    
    """ Save the checkpoint"""
    def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str, best_acc=0):
        """
        Saves model and optimizer state_dict to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }, path)
        logger.info(f"Checkpoint saved at {path}")
    
    val_acc_max = 0.0

    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []

    loss_epochs = []
    train_epochs = []

    train_loader, val_loader = get_loader(batch_size, data_dir, json_list, fold, roi)

    for epoch in range(start_epoch, max_epochs):

        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"{time.ctime()} - Epoch {epoch}")
        epoch_time = time.time()

        train_loss =  train_epoch(
            model,

            train_loader,

            optimizer,

            epoch,

            loss_func,
            
            batch_size=batch_size,
            max_epochs=max_epochs,
            logger=logger,
            device=device,
        )

        logger.info(f"Final training {epoch + 1}/{max_epochs}")
        logger.info(f"Loss {train_loss:.4f}")
        logger.info(f"Time {time.time() - epoch_time:.2f}s")

        for param_group in optimizer.param_groups:
            logger.info(f"Current learning rate: {param_group['lr']}")


        if (epoch + 1) % val_every == 0 or epoch == 0 or (epoch + 1 == max_epochs):

            loss_epochs.append(train_loss)
            train_epochs.append(int(epoch))

            epoch_time = time.time()

            val_acc = valid_epoch(
                model=model,
                loader=val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
                decollate_batch=decollate_batch,
                max_epochs=max_epochs,
                logger=logger,
                device=device
            )

            dice_et = val_acc[0]
            dice_tc = val_acc[1]
            dice_wt = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            
            logger.info(f"Final validation stats {epoch}/{max_epochs - 1}")
            logger.info(f"dice_tc {dice_tc}")
            logger.info(f"dice_wt {dice_wt}")
            logger.info(f"dice_et {dice_et}")
            logger.info(f"Dice_Avg {val_avg_acc}")
            logger.info(f"Time {time.time() - epoch_time:.2f}")

            
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)


            if val_avg_acc > val_acc_max:
                logger.info(f"new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).")
                val_acc_max = val_avg_acc
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, path=save_checkpoint_path, best_acc=val_acc_max)
            

        scheduler.step()


    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max}")


    return (
        val_acc_max
    )
