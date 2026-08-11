import time
import torch
from monai.data import decollate_batch
from .AverageMeter import AverageMeter


def val_epoch(
    device,
    max_epochs,
    model,
    loader,
    epoch,
    acc_func,
    model_inferer,
    post_label,
    post_pred,
    logger,
    amp_enabled=False,
    amp_dtype=torch.bfloat16,
    class_names=None,
):

    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        if len(loader) == 0:
            logger.info("[WARNING] No validation examples found!")

        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model_inferer(data)
            logits = logits.float()
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)

            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_label_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_label_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            names = class_names or [f"c{i}" for i in range(len(run_acc.avg))]

            if ((idx + 1) % 30 == 0):
                logger.info(f"Val {epoch}/{max_epochs} {idx}/{len(loader)}")
                for name, value in zip(names, run_acc.avg):
                    logger.info(f"dice_{name} {value}")
                logger.info(f"Time {time.time() - start_time:.2f}")

            start_time = time.time()

    return run_acc.avg
