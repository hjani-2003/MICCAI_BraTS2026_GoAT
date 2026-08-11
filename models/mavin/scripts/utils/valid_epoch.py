import time

import torch

from .meters import AverageMeter
 
def valid_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer,
    post_sigmoid,
    post_pred,

    decollate_batch,
    max_epochs,
    logger,
    device,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        if len(loader) == 0:
            logger.info("[WARNING] No validation examples found!")

        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)

            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            dice_et = run_acc.avg[0]
            dice_tc = run_acc.avg[1]
            dice_wt = run_acc.avg[2]
            
            if ((idx + 1) % 30 == 0):
                logger.info(f"Val {epoch}/{max_epochs} {idx}/{len(loader)}")
                logger.info(f"dice_tc {dice_tc}")
                logger.info(f"dice_wt {dice_wt}")
                logger.info(f"dice_et {dice_et}")
                logger.info(f"Time {time.time() - start_time:.2f}")

            start_time = time.time()

    return run_acc.avg
