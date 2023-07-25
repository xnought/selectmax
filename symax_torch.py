import torch


def symax(x, dim=0, eta=1):
    sizes = torch.abs(x)
    return sizes / (eta + sizes.sum(dim=dim))
