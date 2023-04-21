fixed_params = {
    'max_z': 20,
    'wd': 0,
    'lin1_tailact': True,
    'ln_lin1': True,
    'ln_s2v': True,
    'batch_size': 16,
    'lr': 0.001,
    'warmup': 100,
    'rbf': 'nexpnorm',
    'ratio_y': 0.003,
    'patience': 120,
    'initlr_ratio': 0.01,
    'add_ef2dir': True,
    'cutoff': 7.5,
    'dir2mask_tailact': True,
    'ef2mask_tailact': False,
    'ef_decay': True,
    'ef_dim': 32,
    'ev_decay': True,
    'hid_dim': 256,
    'ln_emb': False,
    'minlr_ratio': 0.01,
    'num_mplayer': 6,
    'rbound_lower': 0.0,
    "use_dir1": False
}

diff_params = {
    'aspirin': {},
    "benzene": {
        'add_ef2dir': False,
        'cutoff': 11.5,
        'ln_emb': True,
        'num_mplayer': 4,
        'patience': 30,
        'ratio_y': 0.062,
        'rbound_lower': 0.3,
        'initlr_ratio': 1,
        "warmup": -1,
        'batch_size': 32
    },
    "uracil": {},
    'salicylic_acid': {
        'ev_decay': False,
        'cutoff': 8.0,
        'ln_emb': True,
        'num_mplayer': 4,
        'dir2mask_tailact': False,
        'ratio_y': 0.001
    },
    'malonaldehyde': {
        'num_mplayer': 3,
        'ln_emb': True,
        'ef_decay': False,
        'ratio_y': 0.013,
        'dir2mask_tailact': False,
        'cutoff': 6.5
    },
    'ethanol': {
        'cutoff': 5.5,
        'ln_emb': True,
        'ef_decay': False,
        'add_ef2dir': False,
        'ratio_y': 0.025,
        'num_mplayer': 4,
        'dir2mask_tailact': False
    },
    'toluene': {
        'num_mplayer': 5,
        'add_ef2dir': False,
        'cutoff': 7.0,
        'dir2mask_tailact': False,
        'ratio_y': 0.001
    },
    'naphthalene': {
        'ev_decay': False,
        'cutoff': 8.0,
        'ln_emb': True,
        'ef_decay': False,
        'num_mplayer': 5,
        'dir2mask_tailact': False,
        'ratio_y': 0.001
    }
}


def get_md17_params(dataset):
    tp = fixed_params.copy()
    tp.update(diff_params[dataset])
    return tp