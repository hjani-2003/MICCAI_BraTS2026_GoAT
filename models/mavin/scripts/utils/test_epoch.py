import time
import torch
from monai.metrics import HausdorffDistanceMetric

from .meters import AverageMeter

def test_epoch(
    device,
    model,
    loader,
    epoch,
    acc_func,
    model_inferer,
    post_sigmoid,
    post_pred,
    logger,
    decollate_batch
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    # Hausdorff meters (for 3 regions)
    hd_meter = AverageMeter()

    # MONAI Hausdorff 95% metric
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,
        reduction="none"
    )

    with torch.no_grad():
        if len(loader) == 0:
            logger.info("No testing examples found!")

        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].to(device)
            target = batch_data["label"].to(device)

            logits = model_inferer(data)

            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)

            val_output_convert = [
                post_pred(post_sigmoid(v)) for v in val_outputs_list
            ]

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            hd_metric.reset()

            preds_bin = torch.stack(val_output_convert).to(device)
            labels_bin = torch.stack(val_labels_list).to(device)

            hd_metric(y_pred=preds_bin, y=labels_bin)

            hd = hd_metric.aggregate()   # shape: (B, C) or (C,) depending

            # Make sure it's (C,)
            if hd.dim() == 2:
                hd = hd.mean(dim=0)

            # Replace NaNs
            hd = torch.nan_to_num(hd, nan=0.0)

            hd_meter.update(hd.cpu().numpy(), n=1)

            dice_et, dice_tc, dice_wt = run_acc.avg
            hd_et, hd_tc, hd_wt = hd_meter.avg

            logger.info(f"Test {epoch}/1 {idx}/{len(loader)}")
            logger.info(f"DICE ET:{dice_et:.4f}  TC:{dice_tc:.4f}  WT:{dice_wt:.4f}")
            logger.info(f"HD95 ET:{hd_et:.4f} TC:{hd_tc:.4f} WT:{hd_wt:.4f}")
            logger.info(f"Time {time.time() - start_time:.2f}")

            start_time = time.time()

    return run_acc.avg, hd_meter.avg