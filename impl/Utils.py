from typing import Tuple
import torch
from torch import Tensor
from typing import Tuple, Iterable
import numpy as np

@torch.jit.script
def innerprod(v1: Tensor, v2: Tensor) -> Tensor:
    return (v1 * v2).sum(dim=3)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

def train_grad(data, optimizer, mod, loss_fn, ratio_y, ratio_dy):
    mod.train()
    optimizer.zero_grad()
    z, pos, y, dy = data
    pos.requires_grad_(True)
    pred = mod(z, pos)
    dpred = -torch.autograd.grad(
        [torch.sum(pred)],
        [pos],
        create_graph=True,
        retain_graph=True,
    )[0]
    loss_y = loss_fn(pred, y)
    loss_dy = loss_fn(dpred, dy)
    loss = ratio_y * loss_y + ratio_dy * loss_dy
    loss.backward()
    optimizer.step()
    return loss_y.item(), loss_dy.item()


def test_grad(data, mod, score_fn):
    mod.eval()
    z, pos, y, dy = data
    pos.requires_grad_(True)
    pred = mod(z, pos)
    dpred = -torch.autograd.grad(
        [torch.sum(pred)],
        [pos],
        create_graph=False,
        retain_graph=False,
    )[0]
    loss_y = score_fn(pred, y)
    loss_dy = score_fn(dpred, dy)
    return loss_y.item(), loss_dy.item()


def train(data, optimizer, mod, loss_fn):
    mod.train()
    optimizer.zero_grad()
    z, pos, y = data
    pred = mod(z, pos)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def traindl(dl, optimizer, mod, loss_fn):
    mod.train()
    losss = []
    for batch in dl:
        optimizer.zero_grad()
        z, pos, y = batch
        pred = mod(z, pos)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losss.append(loss.item())
    return np.average(losss) 


@torch.no_grad()
def test(data, mod, score_fn):
    mod.eval()
    z, pos, y = data
    pred = mod(z, pos)
    loss_y = score_fn(pred, y)
    return loss_y.item()


@torch.no_grad()
def testdl(dl, mod, score_fn):
    mod.eval()
    sizes = []
    losss = []
    for batch in dl:
        z, pos, y = batch
        sizes.append(z.shape[0])
        pred = mod(z, pos)
        loss_y = score_fn(pred, y)
        losss.append(loss_y.item())
    losss = np.array(losss)
    sizes = np.array(sizes)
    return np.sum(losss * sizes) / np.sum(sizes)



class tensorDataloader:

    def __init__(self,
                 tensors: Iterable[Tensor],
                 batch_size: int,
                 droplast: bool,
                 device: torch.DeviceObjType,
                 shuffle: bool = True):
        self.tensors = tuple(_.contiguous().to(device) for _ in tensors)
        self.device = device
        self.droplast = droplast
        lens = np.array([_.shape[0] for _ in self.tensors])
        assert np.all((lens - lens[0]) == 0), "tensors must have same size"
        self.len = lens[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        ret = self.len / self.batch_size
        if self.droplast:
            ret = np.floor(ret)
        else:
            ret = np.ceil(ret)
        return ret

    def __iter__(self):
        if self.shuffle:
            self.perm = torch.randperm(self.len, device=self.device)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx + self.batch_size >= self.len and self.droplast:
            raise StopIteration
        if self.idx >= self.len:
            raise StopIteration
        if self.shuffle:
            slice = self.perm[self.idx:self.idx + self.batch_size]
            self.idx += self.batch_size
            return tuple(_[slice] for _ in self.tensors)
        else:
            self.idx += self.batch_size
            return tuple(_[self.idx - self.batch_size:self.idx]
                         for _ in self.tensors)
