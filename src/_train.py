
import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net
from _utils import format_time

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

def main(cfg):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    sampler= DistributedSampler(
        train_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler= sampler,
        batch_size= cfg.batch_size,
        num_workers= 4,
    )

    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler= DistributedSampler(
        valid_ds,
        num_replicas=cfg.world_size,
        rank=cfg.local_rank,
        shuffle=False
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler= sampler,
        batch_size= cfg.batch_size_val,
        num_workers= 4,
    )

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone)
    model= model.to(cfg.local_rank)
    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model,
            decay=cfg.ema_decay,
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    model= DistributedDataParallel(
        model,
        device_ids=[cfg.local_rank],
        )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)

    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart= time.time()
            train_dl.sampler.set_epoch(epoch)

            # Train loop
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)

                with autocast(cfg.device.type):
                    logits = model(x)

                loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss.append(loss.item())

                if ema_model is not None:
                    ema_model.update(model)

                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch,
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1,
                        len(train_dl)+1,
                    ))

        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)

                with autocast(cfg.device.type):
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)

            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()

        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')

                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)

        # Exits training on all ranks
        dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return



if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _, total = torch.cuda.mem_get_info(device=rank)

    # Init
    setup(rank, world_size)
    time.sleep(rank)
    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
    time.sleep(world_size - rank)

    # Seed
    set_seed(cfg.seed+rank)

    # Run
    cfg.local_rank= rank
    cfg.world_size= world_size
    main(cfg)
    cleanup()
