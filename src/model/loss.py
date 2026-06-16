import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weight(train_loader, num_classes=86, clip_min=0.2, clip_max=5.0):
    counts = torch.zeros(num_classes, dtype=torch.long)

    for batch in train_loader:
        h, t, r, y = batch
        y = y.cpu().long()
        counts += torch.bincount(y, minlength=num_classes)

    counts_np = counts.numpy()
    total = counts_np.sum()

    weights = total / (num_classes * np.maximum(counts_np, 1))
    weights = weights / weights.mean()
    weights = np.clip(weights, clip_min, clip_max)

    return torch.tensor(weights, dtype=torch.float32), counts


def make_ce_loss(args, device, train_loader=None, num_classes=86):
    weight = None
    # weight, counts = compute_class_weight(
    #     train_loader,
    #     num_classes=num_classes
    # )
    # weight = weight.to(device)

    return nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=0.0,
    )

def info_nce_symmetric(u2d, u3d, temperature=0.1):
    # u2d, u3d normalized: (m, d)
    labels = torch.arange(u2d.size(0), device=u2d.device)
    logits_23 = (u2d @ u3d.t()) / temperature
    logits_32 = (u3d @ u2d.t()) / temperature
    return F.cross_entropy(logits_23, labels) + F.cross_entropy(logits_32, labels)