import time
import torch
from monai.data import decollate_batch
from .AverageMeter import AverageMeter


def val_epoch(device, max_epochs, model, loader, epoch, acc_func, model_inferer, post_softmax, post_pred, post_label, logger):
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

            val_output_convert = [post_pred(post_softmax(t)) for t in val_outputs_list]
            val_label_convert = [post_label(t) for t in val_labels_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_label_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if (idx + 1) % 30 == 0:
                dice_bg, dice_ncr, dice_ed, dice_et = run_acc.avg
                logger.info(f"Val {epoch}/{max_epochs} {idx}/{len(loader)}")
                logger.info(f"dice_bg {dice_bg:.4f}  dice_ncr {dice_ncr:.4f}  dice_ed {dice_ed:.4f}  dice_et {dice_et:.4f}")
                logger.info(f"Time {time.time() - start_time:.2f}s")

            start_time = time.time()

    return run_acc.avg
