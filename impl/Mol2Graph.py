from typing import Final, Tuple
import torch.nn as nn
from torch import Tensor
import torch
from .ModUtils import CosineCutoff, Imod
from .Rbf import rbf_class_mapping

EPS = 1e-6


class EfDecay(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ea, ef):
        return ea, ea.unsqueeze(-1) * ef


class Mol2Graph(nn.Module):
    '''
    Convert a molecule to a graph that GNNLF can process
    '''
    def __init__(self,
                 hid_dim: int,
                 ef_dim: int,
                 rbf: str,
                 cutoff: float,
                 max_z: int,
                 ef_decay: bool = False,
                 ln_emb: bool = False,
                 **kwargs):
        super().__init__()
        self.cutoff_fn = CosineCutoff(cutoff)
        self.rbf_fn = rbf_class_mapping[rbf](ef_dim, cutoff, **kwargs)
        self.z_emb1 = nn.Embedding(max_z + 1, hid_dim, padding_idx=0)
        self.ef_decay = EfDecay() if ef_decay else Imod()
        self.ln_emb = nn.LayerNorm(hid_dim,
                                   elementwise_affine=kwargs['ln_learnable']
                                   ) if ln_emb else nn.Identity()

    def forward(self, z: Tensor, pos: Tensor):
        '''
        z (B, N)
        pos (B, N, 3)
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        '''
        EPS = 1e-6
        B, N = z.shape[0], z.shape[1]
        s = self.z_emb1(z)
        s = self.ln_emb(s)
        ev = pos.unsqueeze(2) - pos.unsqueeze(1)
        idx = torch.arange(N, device=s.device)
        ev[:, idx, idx] = 1  # avoid ev=0. norm backward produce None
        el = torch.linalg.vector_norm(ev, dim=-1)
        el = el.clone()
        ev = ev / (el.unsqueeze(-1) + EPS)

        el[:, idx, idx] = 0  # remove self_loop
        ev[:, idx, idx] = 0

        ef = self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1)
        ea = self.cutoff_fn(el)
        mask = (z == 0)
        mask = (mask).unsqueeze(2) + (mask).unsqueeze(1)
        ea[mask] = 0
        ea, ef = self.ef_decay(ea, ef)
        return s, ea, ev, ef