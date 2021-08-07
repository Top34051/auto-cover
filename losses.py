import torch
from torch import nn


def adv_loss(tensor, real=True):
    if real:
        return torch.mean(torch.clamp(1 - tensor, min=0))
    else:
        return torch.mean(torch.clamp(1 + tensor, min=0))


def embed_loss(a1, a2, ab1, ab2):
    return torch.mean(((a1 - a2) - (ab1 - ab2)) ** 2) - nn.MSELoss()(a1, a2) * nn.MSELoss()(ab1, ab2)


def margin_loss(a1, a2, delta=2.0):
    logits = torch.sqrt((a1 - a2) ** 2)
    return torch.mean(torch.clamp(delta - logits, min=0))