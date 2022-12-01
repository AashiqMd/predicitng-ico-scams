
import logging
import torch
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_data
import torch_model


log = logging.getLogger(__name__)


def get_device(use_cuda):
    device = "cpu"
    if use_cuda:
        if torch.cuda.is_available():
            device = "cuda"
            log.info("gpu found, using CUDA")
        else:
            log.info("no gpu found, defaulting to cpu")
    return torch.device(device)


def main(flags):
    torch.manual_seed(flags.random_seed)
    device = get_device(flags.cuda)

    train_dataset, valid_dataset = torch_data.get_dataset(flags.task, flags.split, flags.random_seed)
    train_loader =  DataLoader(train_dataset, batch_size=flags.bs, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=flags.bs, shuffle=False, num_workers=1)
        

    model = torch_model.get_model(flags)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if flags.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=flags.lr, momentum=flags.momentum, weight_decay=flags.weight_decay)
    elif flags.optim == 'adam':
        optimizer = torch.optim.AdamW(params, lr=flags.lr, weight_decay=flags.weight_decay)
    else:
        raise RuntimeError(f'did not recognize optimizer {flags.optim}')

    if flags.scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif flags.scheduler == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    else:
        raise RuntimeError(f'did not recognize scheduler {flags.scheduler}')
    
    weight = None
    if flags.weight_classes:
        num_pos = sum(ex[1] == 1 for ex in train_dataset)
        weight = (len(train_dataset) - num_pos) / num_pos * flags.pos_weight_mod
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    model.train()
    for epoch in range(1, flags.num_epochs + 1):
        model.train()
        for xs, ys in tqdm(train_loader, file=sys.stdout, desc=f"Training epoch {epoch}"):
            xs, ys = xs.to(device), ys.to(device)
            scores = model(xs)
            loss = criterion(scores, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % flags.valid_every_n == 0 or epoch == flags.num_epochs:
            valid_loss, valid_acc, true_pos, p_denom, r_denom = 0, 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                for xs, ys in valid_loader:
                    xs, ys = xs.to(device), ys.to(device)
                    scores = model(xs)
                    loss = criterion(scores, ys)

                    preds = torch.sigmoid(scores).round()
                    acc = (preds == ys).sum().float()

                    tp = ys.masked_select(preds == 1).sum()
                    true_pos += tp
                    p_denom += (preds == 1).sum()
                    r_denom += (ys == 1).sum()

                    valid_loss += loss
                    valid_acc += acc
            valid_loss /= len(valid_dataset)
            valid_acc /= len(valid_dataset)
            valid_p = true_pos / p_denom
            valid_r = true_pos / r_denom

            log.info("Validation stats:"
                    f"loss: {valid_loss:.3f}, "
                    f"accuracy: {valid_acc:.3f}, "
                    f"precision: {valid_p:.3f}, "
                    f"recall: {valid_r:.3f}, "
                    f"f1: {2 * (valid_p * valid_r) / (valid_p + valid_r):.3f}"
            )
            lr_scheduler.step(valid_loss)
    print("finished training")
