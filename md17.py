'''
Copied from https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/datasets/md17.py
'''
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np


class myMD17(InMemoryDataset):

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="aspirin_dft.npz",
        benzene="benzene2017_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, molecule, transform=None, pre_transform=None):

        self.molecule = molecule
        super(myMD17, self).__init__(root, transform, pre_transform)
        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [myMD17.molecule_files[self.molecule]]

    @property
    def processed_file_names(self):
        return [f"md17-{self.molecule}.pt"]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(myMD17.raw_url + file_name, self.raw_dir)

    def process(self):
        path = self.raw_paths[0]
        data_npz = np.load(path)
        z = torch.from_numpy(data_npz["z"]).long()
        positions = torch.from_numpy(data_npz["R"]).float()
        energies = torch.from_numpy(data_npz["E"])
        forces = torch.from_numpy(data_npz["F"]).float()

        samples = []
        for pos, y, dy in zip(positions, energies, forces):
            samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
