# Python native libraries
import pdb
import os
import itertools
import joblib
from timeit import default_timer as timer
from pathlib import Path

# Science libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Other libraries
from twilio.rest import Client
from tqdm import tqdm

# Project libraries
import training
import scaling
import data
import keys
import models
from settings import Settings as S

# No more scientific notation
np.set_printoptions(suppress=True, linewidth=np.inf)

if __name__ == '__main__':
    # %% Check if we forgot to update the experiment number
    # assert not os.path.exists(f'plots/lstm_exp_{EXPERIMENT_N}.png'), 'Experiment number already exists'
    # assert REDUCE_LR_PATIENCE < EARLY_STOP_PATIENCE, 'Training will stop early before reducing LR'

    # %% Twilio set-up
    client = Client(keys.account_sid, keys.auth_token)

    # %% Scaler set-up
    scaler = scaling.load_or_create(S.SCALER_PATH, S.SCALER_CREATION_DIRS, S.INPUTS, S.OUTPUTS)
    
    # %% Data set-up
    train_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=S.LOOKBACK, inputs=S.INPUTS, outputs=S.OUTPUTS, scaler=scaler, start=1, end=510+1)
    train_loader = DataLoader(train_dataset, batch_size=S.BATCH_SIZE, shuffle=True, num_workers=S.N_WORKERS)

    val_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=S.LOOKBACK, inputs=S.INPUTS, outputs=S.OUTPUTS, scaler=scaler, start=510, end=638+1)
    val_loader = DataLoader(val_dataset, batch_size=S.BATCH_SIZE, shuffle=True, num_workers=S.N_WORKERS)
    
    # %% CUDA set-up
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'CUDA is {is_cuda}')

    # %% Model set-up
    model = models.PINNS(seq_len=S.LOOKBACK, n_inputs=len(S.INPUTS), n_lstm_layers=2, lstm_activations=128, lstm_td_activations=10, dense_activations=10, n_dense_layers=5, use_lstm=S.USE_LSTM, use_pinns=S.USE_PINNS)
    model.to(device)
    print(f'{model} | lstm={model.use_lstm} | pinns={model.use_pinns}')

    # loss_fn = nn.MSELoss()
    adam_opt = optim.Adam(model.parameters(), eps=1e-07)
    lbfgs_opt = optim.LBFGS(model.parameters(), max_iter=50000)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam_opt, patience=REDUCE_LR_PATIENCE)

    # %% Training loop
    training.train(device, model, adam_opt, 200, train_loader, val_loader, client)
    training.train(device, model, lbfgs_opt, 100, train_loader, val_loader, client)