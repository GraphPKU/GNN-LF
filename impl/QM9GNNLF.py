# GNN-LF for QM9 dataset. Nearly the same as GNNLF.py
from typing import Final
import torch.nn as nn
import torch
from torch.nn import ModuleList
from .Mol2Graph import Mol2Graph
from .Utils import innerprod
from .OutputModule import output_dict


class GNNLF(torch.nn.Module):
    ev_decay: Final[bool]
    add_ef2dir: Final[bool]
    use_dir1: Final[bool]
    use_dir2: Final[bool]
    use_dir3: Final[bool]  
    def __init__(self,
                 hid_dim: int,
                 num_mplayer: int,
                 ef_dim: int,
                 y_mean: float,
                 y_std: float,
                 global_y_mean: float,
                 ev_decay: bool,
                 add_ef2dir: bool,
                 use_dir1: bool = False,
                 use_dir2: bool = True,
                 use_dir3: bool = True,
                 tar: str = "scalar",
                 **kwargs):
        super().__init__()
        self.add_ef2dir = add_ef2dir
        kwargs["ln_learnable"] = False
        kwargs["act"] = nn.SiLU(inplace=True)
        self.mol2graph = Mol2Graph(hid_dim, ef_dim, **kwargs)

        self.neighbor_emb = NeighborEmb(hid_dim, **kwargs)
        self.s2v = CFConvS2V(hid_dim, **kwargs)
        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.interactions = ModuleList(
            [DirCFConv(hid_dim, **kwargs) for _ in range(num_mplayer)])
        self.output_module = output_dict[tar](hid_dim, **kwargs)
        self.ef_proj = nn.Sequential(
            nn.Linear(ef_dim, hid_dim),
            nn.SiLU(inplace=True) if kwargs["ef2mask_tailact"] else nn.Identity())
        dir_dim = hid_dim * (use_dir1 + use_dir2 +
                               use_dir3) + self.add_ef2dir * ef_dim
        self.dir_proj = nn.Sequential(
            nn.Linear(dir_dim, dir_dim), nn.SiLU(inplace=True),
            nn.Linear(dir_dim, hid_dim),
            nn.SiLU(inplace=True) if kwargs["dir2mask_tailact"] else nn.Identity())
        self.ev_decay = ev_decay
        self.use_dir1 = use_dir1
        self.use_dir2 = use_dir2
        self.use_dir3 = use_dir3

    def forward(self, z, pos):
        s, ea, ev, ef = self.mol2graph(z, pos)
        mask = self.ef_proj(ef) * ea.unsqueeze(-1)
        s = self.neighbor_emb(z, s, mask)
        v = self.s2v(s, ev, mask)
        dirs = []
        if self.use_dir1:
            dir1 = innerprod(v.unsqueeze(1), ev.unsqueeze(-1))
            dirs.append(dir1)
        if self.use_dir2:
            dir2 = innerprod(v.unsqueeze(2), ev.unsqueeze(-1))
            dirs.append(dir2)
        if self.use_dir3:
            dir3 = innerprod(
                self.q_proj(v).unsqueeze(1),
                self.k_proj(v).unsqueeze(2))
            dirs.append(dir3)
        dir = torch.cat(dirs, dim=-1)
        if self.ev_decay:
            dir = dir * ea.unsqueeze(-1)
        if self.add_ef2dir:
            dir = torch.cat((dir, ef), dim=-1)
        dir = self.dir_proj(dir)
        mask = mask * dir
        for interaction in self.interactions:
            s = interaction(s, mask) + s
        s = self.output_module(z, s, pos)
        return s


class CFConv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, s, mask):
        '''
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        '''
        s = mask * s.unsqueeze(1)
        s = torch.sum(s, dim=2)
        return s


class DirCFConv(nn.Module):

    def __init__(self, hid_dim: int, ln_lin1: bool, lin1_tailact: bool,
                 **kwargs):
        super().__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"])
            if ln_lin1 else nn.Identity(),
            kwargs["act"] if lin1_tailact else nn.Identity())

    def forward(self, s, ef_mask):
        '''
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        '''
        s = torch.sum(ef_mask * self.lin1(s).unsqueeze(1), dim=2)
        return s


class NeighborEmb(nn.Module):

    def __init__(self, hid_dim: int, max_z: int, ln_emb: bool, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hid_dim, padding_idx=0)
        self.conv = CFConv()
        self.ln_emb = nn.LayerNorm(hid_dim,
                                   elementwise_affine=kwargs['ln_learnable']
                                   ) if ln_emb else nn.Identity()

    def forward(self, z, s, mask):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.conv(s_neighbors, mask)
        s = s + s_neighbors
        return s


class CFConvS2V(nn.Module):

    def __init__(self, hid_dim: int,  ln_s2v: bool,
                 lin1_tailact: bool, **kwargs):
        super().__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=kwargs['ln_learnable'])
            if ln_s2v else nn.Identity(),
            kwargs["act"] if lin1_tailact else nn.Identity())

    def forward(self, s, ev, mask):
        '''
        s (B, N, hid_dim)
        v (B, N, 3, hid_dim)
        ea (B, N, N)
        ef (B, N, N, ef_dim)
        ev (B, N, N, 3)
        '''
        s = self.lin1(s)
        s = s.unsqueeze(1) * mask
        v = s.unsqueeze(3) * ev.unsqueeze(-1)
        v = torch.sum(v, dim=2)
        return v
