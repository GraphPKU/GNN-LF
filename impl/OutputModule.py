'''
output modules for different qm9 targets
'''
import torch.nn as nn
import torch
import ase


class QM9scalar(nn.Module):
    def __init__(self, hid_dim, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"]), nn.Linear(hid_dim, 1, bias=False))
        self.atom_ref = nn.Embedding(kwargs["max_z"], 1, padding_idx=0)
    
    def forward(self, z, s, pos):
        s = self.mlp(s) + self.atom_ref(z)
        s[z==0] = 0
        s = torch.sum(s, dim=1)
        return s

class QM9dipole_moment(nn.Module):
    def __init__(self, hid_dim:int, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"]), nn.Linear(hid_dim, 1, bias=False))
    
    def forward(self, z, s, pos):
        '''
        for neutral molecule, r doesn't need to minus the position of mass center
        '''
        mask = (z==0)
        q = self.mlp(s)
        q[mask] = 0
        q = q - torch.sum(q, dim=1, keepdim=True)/torch.sum(1-mask.to(torch.float), dim=1, keepdim=True).unsqueeze_(-1)
        q[mask] = 0
        ret = torch.sum(q * pos, dim=1)
        ret = torch.norm(ret, dim=-1, keepdim=True)
        return ret


class QM9electronic_spatial_extent(nn.Module):
    def __init__(self, hid_dim:int, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(hid_dim, elementwise_affine=kwargs["ln_learnable"]), nn.Linear(hid_dim, 1, bias=False))
        self.atom_ref = nn.Embedding(kwargs["max_z"], 1, padding_idx=0)
        self.act = nn.Softplus()
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        atomic_mass[0] = 0
        self.register_buffer("atomic_mass", atomic_mass)
    
    def forward(self, z, s, pos):
        mask = (z==0)
        q = self.act(self.mlp(s) + self.atom_ref(z))
        q[mask] = 0
        mass = self.atomic_mass[z].unsqueeze(-1)
        c = torch.sum(mass * pos, dim=1, keepdim=True) / torch.sum(mass, dim=1, keepdim=True)
        ret = torch.sum(q.squeeze(-1) * torch.sum(torch.square(pos-c), dim=-1), dim=1, keepdim=True)
        return ret


output_dict = {"scalar": QM9scalar, "dipole_moment": QM9dipole_moment, "electronic_spatial_extent": QM9electronic_spatial_extent}