from torch.optim import Adam
import torch
from Dataset import load
import argparse
from torch.nn.functional import l1_loss, mse_loss
from impl.QM9GNNLF import GNNLF
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from impl import Utils
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, random_split



def buildModel(**kwargs):
    if args.dataset == "dipole_moment":
        tar = "dipole_moment"
    elif args.dataset == "electronic_spatial_extent":
        tar = "electronic_spatial_extent"
    else:
        tar = "scalar"
    mod = GNNLF(y_mean=y_mean,
                    y_std=y_std,
                    global_y_mean=global_y_mean,
                    tar=tar,
                    **kwargs)
    print(f"numel {sum(p.numel() for p in mod.parameters() if p.requires_grad)}")
    return mod


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="dipole_moment", help="target in qm9 dataset")
parser.add_argument('--modname', type=str, default="0", help="filename used to save model")
parser.add_argument('--repeat', type=int, default=3, help="number of repeated runs")
args = parser.parse_args()

device = torch.device("cuda")
dataset = load(args.dataset)
if args.dataset in [
        "dipole_moment", "isotropic_polarizability", "homo", "lumo", "gap",
        "electronic_spatial_extent", "zpve", "energy_U0", "energy_U",
        "enthalpy_H", "free_energy", "heat_capacity"
]:
    ratio = [110000, 10000, 10831]
else:
    raise NotImplementedError

N = dataset["z"].shape[1]
ds = TensorDataset(dataset['z'].reshape(-1,
                                        N), dataset['pos'].reshape(-1, N, 3),
                   dataset['y'].reshape(-1, 1))
y_mean = None
y_std = None
global_y_mean = 0.0


def work(total_step: int = 4000,
         batch_size: int = 256,
         save_model: bool = False,
         do_test: bool = False,
         jump_train: bool = False,
         max_early_stop: int = 100,
         lr: float = 1e-3,
         warmup: int = 3,
         patience: int = 10,
         **kwargs):
    global y_mean, y_std
    NAN_PANITY = 1e3
    trn_ds, val_ds, tst_ds = random_split(ds, ratio)
    trn_dl = Utils.tensorDataloader(trn_ds.dataset[trn_ds.indices], batch_size,
                                    True, device, True)
    val_dl = Utils.tensorDataloader(val_ds.dataset[val_ds.indices], 4*batch_size,
                                    False, device, False)
    tst_dl = Utils.tensorDataloader(tst_ds.dataset[tst_ds.indices], 4*batch_size,
                                    False, device, False)
    y_mean = torch.mean(trn_ds.dataset[trn_ds.indices][2]).item()
    y_std = torch.std(trn_ds.dataset[trn_ds.indices][2]).item()
    mod = buildModel(**kwargs).to(device)
    best_val_loss = float("inf")
    if not jump_train:
        opt = Adam(mod.parameters(), lr=lr / 100)
        scd1 = StepLR(opt,
                      1,
                      gamma=100**(1 / (warmup * (110000 // batch_size))))
        scd2 = ReduceLROnPlateau(opt,
                                 mode="min",
                                 factor=0.6,
                                 patience=patience,
                                 min_lr=1e-6)
        early_stop = 0
        for epoch in range(total_step):
            curlr = opt.param_groups[0]["lr"]
            trn_losss = []
            for batch in trn_dl:
                trn_loss_y = Utils.train(batch, opt, mod, mse_loss)
                if np.isnan(trn_loss_y):
                    return NAN_PANITY
                if epoch < warmup:
                    scd1.step()
                trn_losss.append(trn_loss_y)
            trn_loss_y = np.average(trn_losss)

            val_loss = Utils.testdl(val_dl, mod, l1_loss)
            if epoch > warmup:
                scd2.step(val_loss)
            early_stop += 1
            if np.isnan(val_loss):
                return NAN_PANITY
            if val_loss < best_val_loss:
                early_stop = 0
                best_val_loss = val_loss
                if save_model:
                    torch.save(mod.state_dict(), modfilename)
                tst_score = Utils.testdl(tst_dl, mod, l1_loss)
                print(f"tst E {tst_score:.4f}")
            if early_stop > max_early_stop:
                break
            print(
                f"iter {epoch} lr {curlr:.4e} trn E {trn_loss_y:.4f} val E {val_loss:.4f}",
                flush=True)
    
    if do_test:
        mod.load_state_dict(torch.load(modfilename, map_location="cpu"))
        mod = mod.to(device)
        tst_score = Utils.testdl(tst_dl, mod, l1_loss)
        trn_score = Utils.testdl(trn_dl, mod, l1_loss)
        val_score = Utils.testdl(val_dl, mod, l1_loss)
        print(trn_score, val_score, tst_score)
    return val_score

if __name__ == "__main__":
    import qm9_params
    fixed_p = qm9_params.param.copy()
    t1 = time.time()
    import qm9_params
    tp = qm9_params.param
    if args.dataset in ["gap"]:
        tp.update({'num_mplayer': 8, 'lr': 0.003, 'batch_size': 32, 'warmup': 10, 'cutoff': 7.0})
    if args.dataset in ["isotropic_polarizability"]:
        tp.update({'ef_decay': False, 'ev_decay': True, 'lin1_tailact': True, 'dir2mask_tailact': False, 'ef2mask_tailact': True, 'num_mplayer': 7, 'add_ef2dir': True, 'ef_dim': 80, 'ln_emb': False, 'lr': 0.0003, 'batch_size': 32})
    if args.dataset in ["homo", "lumo"]:
        tp.update({'lr': 0.001, 'batch_size': 32, "warmup": 40})
    print(tp)
    for i in range(args.repeat):
        print(f"seed {i}")
        Utils.set_seed(i)
        modfilename = f"save_mod/{args.dataset}.dirschnet.{args.modname}.{i}.pt"
        
        print(
            work(**tp,
             total_step=1000,
             max_early_stop=100,
             save_model=True,
             do_test=True))
    print(f"time {time.time()-t1:.2f} s")
