import time
import numpy as np
import math

from .test_epoch import test_epoch
from .test_dataloader import get_test_loader

""" Define Tester"""
def tester(
    device,
    data_dir,
    json_list,
    model,
    acc_func,
    model_inferer,
    post_sigmoid,
    post_pred,
    logger,
    decollate_batch
):
    # Dice Accuracy
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    
    # Hausdorff Distance 95 percentile
    hd_acc_max = math.inf
    hds_tc = []
    hds_wt = []
    hds_et = []
    hds_avg = []

    for epoch in range(1):
        test_loader = get_test_loader(data_dir, json_list)
        logger.info(f"{time.ctime()} - Epoch {epoch}")
        epoch_time = time.time()

        test_acc_dice, test_acc_hd = test_epoch(
            device=device,
            model=model,
            loader=test_loader,
            epoch=epoch,
            acc_func=acc_func,
            model_inferer=model_inferer,
            post_sigmoid=post_sigmoid,
            post_pred=post_pred,
            logger=logger,
            decollate_batch=decollate_batch
        )
        
        # Dice
        dice_et = test_acc_dice[0]
        dice_tc = test_acc_dice[1]
        dice_wt = test_acc_dice[2]
        val_avg_acc = np.mean(test_acc_dice)
        
        # HD
        hd_et = test_acc_hd[0]
        hd_tc = test_acc_hd[1]
        hd_wt = test_acc_hd[2]
        hd_avg_acc = np.mean(test_acc_hd)

        # Logging
        logger.info(f"Final testing stats {epoch+1}/{1}")
        logger.info(f"dice_tc {dice_tc}")
        logger.info(f"dice_wt {dice_wt}")
        logger.info(f"dice_et {dice_et}")
        logger.info(f"Dice_Avg {val_avg_acc}")
        logger.info(f"=============================")
        logger.info(f"hd_tc {hd_tc}")
        logger.info(f"hd_wt {hd_wt}")
        logger.info(f"hd_et {hd_et}")
        logger.info(f"HD_Avg {hd_avg_acc}")
        logger.info(f"Time {time.time() - epoch_time:.2f}")
        logger.info(f"=============================")
        
        # Dice
        dices_tc.append(dice_tc)
        dices_wt.append(dice_wt)
        dices_et.append(dice_et)
        dices_avg.append(val_avg_acc)
        
        #HD
        hds_tc.append(dice_tc)
        hds_wt.append(dice_wt)
        hds_et.append(dice_et)
        hds_avg.append(hd_avg_acc)

        # if improved (simple check)
        if val_avg_acc > val_acc_max:
            val_acc_max = val_avg_acc
        
        # if improved (simple check)
        if hd_avg_acc < hd_acc_max:
            hd_acc_max = hd_avg_acc

    return (
        val_acc_max,
        hd_acc_max
    )