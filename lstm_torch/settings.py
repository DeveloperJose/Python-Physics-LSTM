import os
from pathlib import Path

class Settings:
    EXPERIMENT_N = 3
    LOOKBACK = 1
    USE_LSTM = False
    USE_PINNS = True

    BATCH_SIZE = 20000
    N_WORKERS = 6
    EARLY_STOP_PATIENCE = 1000

    PLOTS_PATH = Path('plots')
    CHECKPOINTS_PATH = Path('checkpoints')

    INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
    OUTPUTS = ['Vu', 'Vv']
    SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}.pkl')
    SCALER_CREATION_DIRS = ['/home/jperez/data/sled250']