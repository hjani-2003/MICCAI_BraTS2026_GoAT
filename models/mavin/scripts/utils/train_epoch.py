import time

from .meters import AverageMeter
 
def train_epoch(model,
                    
                    loader, 
                    
                    optimizer,
                    
                    epoch, 
                    
                    loss_func,
                    
                    batch_size,
                    max_epochs,
                    logger,
                    device):
    
    model.train()

    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):

        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        run_loss.update(loss.item(), n=batch_size)


        if ((idx + 1) % 30 == 0):
            logger.info(f"Epoch {epoch}/{max_epochs} {idx}/{len(loader)}")
            logger.info(f"Loss {run_loss.avg:.4f}")
            logger.info(f"Time {time.time() - start_time:.2f}")
        
        start_time = time.time()

    return run_loss.avg
