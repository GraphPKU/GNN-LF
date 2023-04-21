# Graph Neural Network With Local Frame for Molecular Potential Energy Surface

This repository is the official implementation of the following paper

Xiyuan Wang, Muhan Zhang: Graph Neural Network With Local Frame for Molecular Potential Energy Surface. LoG 2022.

```
@inproceedings{GNNLF,
  author       = {Xiyuan Wang and
                  Muhan Zhang},
  editor       = {Bastian Rieck and
                  Razvan Pascanu},
  title        = {Graph Neural Network With Local Frame for Molecular Potential Energy
                  Surface},
  booktitle    = {Learning on Graphs Conference, LoG 2022, 9-12 December 2022, Virtual
                  Event},
  series       = {Proceedings of Machine Learning Research},
  volume       = {198},
  pages        = {19},
  publisher    = {{PMLR}},
  year         = {2022}
}
```



#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.11.0](https://pytorch.org/get-started/previous-versions/)

Other required python libraries include: numpy, scikit-learn, optuna, torch_geometric, etc.

#### Prepare Data
We write a script for preparing datasets.
```
python prepare_dataset.py
```

#### Reproduce Results on MD17
To reproduce results of the benzene molecule.
```
python main_md17.py --dataset benzene --test 
```

"benzene" can be replaced with other molecules: benzene, uracil, naphthalene, aspirin, salicylic_acid, malonaldehyde, ethanol, toluene


To do ablation analysis.

NoDir2
```
python main_md17.py --nodir2 --dataset benzene --test 
```

NoDir3
```
python main_md17.py --nodir3 --dataset benzene --test 
```

Global
```
python main_md17.py --global_frame --dataset benzene --test 
```

NoDecomp
```
python main_md17.py --no_filter_decomp --dataset benzene --test 
```

NoShare
```
python main_md17.py --no_share_filter --dataset benzene --test 
```

####  Reproduce Results on QM9
To reproduce results of the homo target
```
python main_qm9.py --dataset homo --test
```
"homo" can be replaced with other targets: dipole_moment, isotropic_polarizability, homo, lumo, gap, electronic_spatial_extent, zpve, energy_U0, energy_U, enthalpy_H, free_energy, heat_capacity
