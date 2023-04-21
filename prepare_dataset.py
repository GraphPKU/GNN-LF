from md17 import myMD17
import qm9 
import torch
for dataset in ['benzene', 'uracil', 'naphthalene', 'aspirin', 'salicylic_acid', 'malonaldehyde', 'ethanol', 'toluene']:
    myMD17("../MD17", dataset)

ds = qm9.QM9("../QM9", None, "dipole_moment")
y = ds.data.y[:, :max(qm9.qm9_target_dict.keys())+1]
lens = [d.z.shape[0] for d in ds]
max_mol_size = max(lens)
poss= []
zs = []
for d in ds:
    poss.append(torch.cat((d.pos, torch.zeros((max_mol_size-d.pos.shape[0], 3), dtype=d.pos.dtype)), dim=0))
    zs.append(torch.cat((d.z, torch.zeros((max_mol_size-d.z.shape[0]), dtype=d.z.dtype)), dim=0))
pos = torch.stack(poss)
z = torch.stack(zs)
torch.save({"pos": pos, "z": z, "y": y}, "../QM9/padded_data.pt")