import time
import torch
from .AverageMeter import AverageMeter

def train_epoch(device,
                max_epochs,
                model,
                loader,
                optimizer,
                epoch,
                loss_func,
                logger,
                amp_enabled=False,
                amp_dtype=torch.bfloat16,
                scaler=None,
                pseudo_lambda=1.0):
    model.train()

    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        is_pseudo = batch_data.get("is_pseudo")

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(data)
            if is_pseudo is None:
                
                loss = loss_func(logits, target)
            else:

                
                is_pseudo = is_pseudo.to(device).view(-1)
                gt_mask = is_pseudo == 0
                pl_mask = is_pseudo == 1
                terms = []
                if gt_mask.any():
                    terms.append(loss_func(logits[gt_mask], target[gt_mask]))
                if pl_mask.any():
                    terms.append(pseudo_lambda * loss_func(logits[pl_mask], target[pl_mask]))
                loss = terms[0]
                for extra in terms[1:]:
                    loss = loss + extra

        if scaler is not None:
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        run_loss.update(loss.item(), n=data.shape[0])

        if ((idx + 1) % 30 == 0):
            logger.info(f"Epoch {epoch}/{max_epochs} {idx}/{len(loader)}")
            logger.info(f"Loss {run_loss.avg:.4f}")
            logger.info(f"Time {time.time() - start_time:.2f}")

        start_time = time.time()

    return run_loss.avg
