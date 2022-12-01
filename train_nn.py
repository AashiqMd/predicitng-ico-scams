
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

    train_dataset, valid_dataset, dictionary = torch_data.get_dataset(flags.task, flags.split, flags.random_seed, flags.tokenizer)
    collate_fn = None
    if dictionary is not None:
        collate_fn = torch_data.pad_collate
    pm = device == torch.device('cuda')
    train_loader =  DataLoader(train_dataset, batch_size=flags.bs, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=pm)
    valid_loader = DataLoader(valid_dataset, batch_size=flags.bs, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=pm)
        

    model = torch_model.get_model(flags, dictionary)
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    
    model.train()
    for epoch in range(1, flags.num_epochs + 1):
        model.train()
        with tqdm(total=len(train_dataset), file=sys.stdout, desc=f"Training epoch {epoch}") as pbar:
            train_loss = 0
            for batch in train_loader:
                xs, ys = batch[0].to(device), batch[1].to(device)
                scores = model(xs)
                loss = criterion(scores, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.update(len(xs))
            train_loss /= len(train_loader)
            pbar.set_description(f"Training epoch {epoch} (ave loss: {train_loss:.5f})")

        if epoch % flags.valid_every_n == 0 or epoch == flags.num_epochs:
            valid_loss, valid_acc, true_pos, p_denom, r_denom = 0, 0, 0, 0, 0
            doc_metrics = {}
            model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    xs, ys = batch[0].to(device), batch[1].to(device)
                    scores = model(xs)
                    loss = criterion(scores, ys)

                    if len(batch) > 2:
                        idxs = batch[2]
                        for bi, idx in enumerate(idxs):
                            idx = idx.item()
                            if idx not in doc_metrics:
                                doc_metrics[idx] = {'cnt': 0, 'score': 0.0, 'label': (ys[bi] == 1).item()}
                            doc_metrics[idx]['cnt'] += 1
                            doc_metrics[idx]['score'] += torch.sigmoid(scores[bi]).item()
                    preds = torch.sigmoid(scores).round()
                    acc = (preds == ys).sum().float()

                    tp = ys.masked_select(preds == 1).sum()
                    true_pos += tp
                    p_denom += (preds == 1).sum()
                    r_denom += (ys == 1).sum()

                    valid_loss += loss.item()
                    valid_acc += acc
            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_dataset)
            valid_p = true_pos / p_denom if p_denom > 0 else 0
            valid_r = true_pos / r_denom
            f1 = 2 * (valid_p * valid_r) / (valid_p + valid_r) if valid_p + valid_r > 0 else 0

            log_string = ("Validation stats:\n"
                f"loss: {valid_loss:.5f}, "
                f"acc: {valid_acc:.3f}, "
                f"prec: {valid_p:.3f}, "
                f"rec: {valid_r:.3f}, "
                f"f1: {f1:.3f}")
            
            if doc_metrics:
                doc_accuracy = 0
                doc_tp = 0
                doc_p_denom = 0
                doc_r_denom = 0
                for doc in doc_metrics.values():
                    pred = doc['score'] / doc['cnt'] > 0.5
                    if pred == doc['label']:
                        doc_accuracy += 1
                        if pred:
                            doc_tp += 1
                    doc_p_denom += pred
                    doc_r_denom += doc['label']
                doc_accuracy /= len(doc_metrics)
                doc_prec = doc_tp / doc_p_denom if doc_p_denom > 0 else 0
                doc_rec = doc_tp / doc_r_denom
                doc_f1 = 2 * (doc_prec * doc_rec) / (doc_prec + doc_rec) if doc_prec + doc_rec > 0 else 0
                log_string += (
                    f", doc_acc: {doc_accuracy:.3f}"
                    f", doc_prec: {doc_prec:.3f}"
                    f", doc_rec: {doc_rec:.3f}"
                    f", doc_f1: {doc_f1:.3f}"
            )
            log.info(log_string)
            lr_scheduler.step(valid_loss)
    print("finished training")

    # TEST CODE FOR GETTING SENTENCE SCORES
    # sentences = []
    # model.eval()
    # with torch.no_grad():
    #     for batch in valid_loader:
    #         xs, ys = batch[0].to(device), batch[1].to(device)
    #         scores = model(xs)
    #         for bi in range(len(xs)):
    #             sentences.append((torch.sigmoid(scores[bi]).item(), (ys[bi] == 1).item(), dictionary.vec2txt(xs[bi], skip_null=True)))
    # sentences.sort()
            
    # import pdb; pdb.set_trace()
