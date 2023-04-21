import torch
from md17 import myMD17
from qm9 import qm9_target_dict
from torch_geometric.datasets.qm9 import atomrefs
from torch_geometric.data import InMemoryDataset
from typing import List, Tuple, Union


def load(name: str) -> InMemoryDataset:
    if name in [
            'aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic_acid','toluene', 'uracil'
    ]:
        return myMD17("../MD17", name)
    elif name in ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]:
        data = torch.load("../QM9/padded_data.pt", map_location="cpu")
        rev_dict = {qm9_target_dict[key]: key for key in qm9_target_dict.keys()}
        idx = rev_dict[name]
        data['y'] = data['y'][:, idx]
        if idx in atomrefs:
            tatomref = torch.zeros(12, dtype=data['y'].dtype)
            for i, z in enumerate([1, 6, 7, 8, 9]):
                tatomref[z] = atomrefs[idx][i]
            taf = tatomref[data['z']].sum(dim=-1)
            data['y'] -= taf
        data['y'] *= 100
        return data
    else:
        raise NotImplementedError(f"No loader for {name}.")


def split(
    dataset: InMemoryDataset, num: List[Union[float, int]]
) -> Tuple[InMemoryDataset, InMemoryDataset, InMemoryDataset]:
    n = len(dataset)
    for i in range(len(num)):
        if isinstance(num[i], float):
            num[i] = round(n * num[i])
    tdataset = dataset.shuffle()
    return tdataset[:num[0]], tdataset[num[0]:num[0] + num[1]], tdataset[num[0] + num[1]:]
