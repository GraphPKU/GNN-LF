from torch.optim import Adam
import torch
from Dataset import load
import argparse
from torch.nn.functional import l1_loss, mse_loss
from impl.GNNLF import GNNLF
from impl.ThreeDimFrame import GNNLF as ThreeDGNNLF
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from impl import Utils
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, random_split


def buildModel(**kwargs):
    if args.threedframe:
        mod = ThreeDGNNLF(y_mean=y_mean,
                          y_std=y_std,
                          global_y_mean=global_y_mean,
                          **kwargs)
    else:
        mod = GNNLF(y_mean=y_mean,
                    y_std=y_std,
                    global_y_mean=global_y_mean,
                    **kwargs)
    print(
        f"numel {sum(p.numel() for p in mod.parameters() if p.requires_grad)}")
    return mod


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="benzene", help="molecule in the md17 dataset")
parser.add_argument('--modname', type=str, default="0", help="filename used to save model")
parser.add_argument('--gemnet_split', action="store_true", help="whether to use the split of gemnet train/val = 1000/1000")
parser.add_argument('--nodir2', action="store_true", help="whether to do ablation study on one kind of coordinate projections")
parser.add_argument('--nodir3', action="store_true", help="whether to do ablation study on frame-frame projections")
parser.add_argument('--global_frame', action="store_true", help="whether to use a global frame rather than a local frame")
parser.add_argument('--no_filter_decomp', action="store_true", help="whether to do ablation study on filter decomposition")
parser.add_argument('--nolin1', action="store_true", help="a hyperparameter")
parser.add_argument('--no_share_filter', action="store_true", help="whether to do ablation study on sharing filters")
parser.add_argument('--cutoff', type=float, default=None, help="cutoff radius")
parser.add_argument('--repeat', type=int, default=3, help="number of repeated runs")
parser.add_argument('--jump_train', action="store_true", help="whether to do test only")
parser.add_argument('--threedframe', action="store_true",  help="whether to do ablation study on frame ensembles")
args = parser.parse_args()

ratio_y = 0.01 # the ratio of energy loss
ratio_dy = 1 # ratio of force loss

device = torch.device("cuda")
dataset = load(args.dataset)
if args.dataset in [
        'benzene', 'uracil', 'naphthalene', 'aspirin', 'salicylic_acid',
        'malonaldehyde', 'ethanol', 'toluene'
]:
    ratio = [950, 50]
else:
    raise NotImplementedError
N = dataset[0].z.shape[0]
global_y_mean = torch.mean(dataset.data.y)
dataset.data.y = (dataset.data.y - global_y_mean).to(torch.float32)
ds = TensorDataset(dataset.data.z.reshape(-1, N),
                   dataset.data.pos.reshape(-1, N, 3),
                   dataset.data.y.reshape(-1, 1),
                   dataset.data.dy.reshape(-1, N, 3))
y_mean = None
y_std = None


def work(lr: float = 1e-3,
         initlr_ratio: float = 1e-1,
         minlr_ratio: float = 1e-3,
         total_step: int = 3000,
         batch_size: int = 32,
         save_model: bool = False,
         do_test: bool = False,
         jump_train: bool = False,
         search_hp: bool = False,
         max_early_stop: int = 500,
         patience: int = 90,
         warmup: int = 30,
         **kwargs):
    global y_mean, y_std, ratio_y, ratio_dy
    if "ratio_y" in kwargs:
        ratio_y = kwargs["ratio_y"]

    NAN_PANITY = 1e1
    if search_hp:
        trn_ds, val_ds, tst_ds = random_split(
            ds, [950, 256, len(ds) - 950 - 256])
    elif args.gemnet_split:
        trn_ds, val_ds, tst_ds = random_split(ds, [1000, 1000, len(ds) - 2000])
    else:
        trn_ds, val_ds, tst_ds = random_split(ds, [950, 50, len(ds) - 1000])
    val_d = next(
        iter(DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)))
    val_d = [_.to(device) for _ in val_d]
    trn_d = next(
        iter(DataLoader(trn_ds, batch_size=len(trn_ds), shuffle=False)))
    trn_d = [_.to(device) for _ in trn_d]
    trn_dl = Utils.tensorDataloader(trn_d, batch_size, True, device)
    y_mean = torch.mean(trn_d[2]).item()
    y_std = torch.std(trn_d[2]).item()
    mod = buildModel(**kwargs).to(device)
    best_val_loss = float("inf")
    if not jump_train:
        opt = Adam(mod.parameters(),
                   lr=lr * initlr_ratio if warmup > 0 else lr)
        scd1 = StepLR(opt,
                      1,
                      gamma=(1 / initlr_ratio)**(1 / (warmup *
                                                      (950 // batch_size)))
                      if warmup > 0 else 1)
        scd = ReduceLROnPlateau(opt,
                                "min",
                                0.8,
                                patience=patience,
                                min_lr=lr * minlr_ratio,
                                threshold=0.0001)
        early_stop = 0
        for epoch in range(total_step):
            curlr = opt.param_groups[0]["lr"]
            trn_losss = [[], []]
            t1 = time.time()
            for batch in trn_dl:
                trn_loss_y, trn_loss_dy = Utils.train_grad(
                    batch, opt, mod, mse_loss, ratio_y, ratio_dy)
                if np.isnan(trn_loss_dy):
                    return NAN_PANITY
                trn_losss[0].append(trn_loss_y)
                trn_losss[1].append(trn_loss_dy)
                if epoch < warmup:
                    scd1.step()
            t1 = time.time() - t1
            trn_loss_y = np.average(trn_losss[0])
            trn_loss_dy = np.average(trn_losss[1])
            val_loss_y, val_loss_dy = Utils.test_grad(val_d, mod, l1_loss)
            val_loss = 0.1 * val_loss_y + val_loss_dy
            early_stop += 1
            scd.step(val_loss)
            if np.isnan(val_loss):
                return NAN_PANITY
            if val_loss < best_val_loss:
                early_stop = 0
                best_val_loss = val_loss
                if save_model:
                    torch.save(mod.state_dict(), modfilename)
            if early_stop > max_early_stop:
                break
            print(
                f"iter {epoch} time {t1} lr {curlr:.4e} trn E {trn_loss_y:.4f} F {trn_loss_dy:.4f} val E {val_loss_y:.4f} F {val_loss_dy:.4f} "
            )
            if epoch % 10 == 0:
                print("", end="", flush=True)
            if trn_loss_dy > 1000:
                return min(best_val_loss, NAN_PANITY)

    if do_test:
        mod.load_state_dict(torch.load(modfilename, map_location="cpu"))
        mod = mod.to(device)
        tst_dl = DataLoader(tst_ds, 1024)
        tst_score = []
        num_mol = []
        for batch in tst_dl:
            num_mol.append(batch[0].shape[0])
            batch = tuple(_.to(device) for _ in batch)
            tst_score.append(Utils.test_grad(batch, mod, l1_loss))
        num_mol = np.array(num_mol)
        tst_score = np.array(tst_score)
        tst_score = np.sum(tst_score *
                           (num_mol.reshape(-1, 1) / num_mol.sum()),
                           axis=0)
        trn_score = Utils.test_grad(trn_d, mod, l1_loss)
        val_score = Utils.test_grad(val_d, mod, l1_loss)
        print(trn_score, val_score, tst_score)
    return min(best_val_loss, NAN_PANITY)


if __name__ == "__main__":

    modfilename = f"save_mod/{args.dataset}.dirschnet.{args.modname}.pt"
    from md17_params import get_md17_params
    tp = get_md17_params(args.dataset)
    tp["use_dir2"] = not args.nodir2
    tp["use_dir3"] = not args.nodir3
    tp["global_frame"] = args.global_frame
    tp["no_filter_decomp"] = args.no_filter_decomp
    tp["nolin1"] = args.nolin1
    tp["no_share_filter"] = args.no_share_filter
    if args.cutoff is not None:
        tp["cutoff"] = args.cutoff
    print(tp)

    for i in range(args.repeat):
        Utils.set_seed(i)
        modfilename = f"save_mod/{args.dataset}.dirschnet.{args.modname}.{i}.pt"
        t1 = time.time()
        print(
            work(**tp,
                total_step=6000,
                max_early_stop=1000,
                save_model=True,
                jump_train=args.jump_train,
                do_test=True))
        print(f"iter {i} time {time.time()-t1:.2f} s")
