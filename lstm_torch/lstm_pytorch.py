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
import scaling
import data
import keys
import models

EXPERIMENT_N = 1
N_EPOCHS = 2

# Model Hyperparameters
BATCH_SIZE = 5024*7
N_WORKERS = 4
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# Dataset Parameters
LOOKBACK = 10
INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
OUTPUTS = ['Vu', 'Vv']
SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}.pkl')
SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled255']

# No more scientific notation
np.set_printoptions(suppress=True, linewidth=np.inf)

if __name__ == '__main__':
    # %% Check if we forgot to update the experiment number
    assert not os.path.exists(f'plots/lstm_exp_{EXPERIMENT_N}.png'), 'Experiment number already exists'
    assert REDUCE_LR_PATIENCE < EARLY_STOP_PATIENCE, 'Training will stop early before reducing LR'

    # %% Twilio set-up
    client = Client(keys.account_sid, keys.auth_token)

    # %% Scaler set-up
    sc = scaling.load_or_create(SCALER_PATH, SCALER_CREATION_DIRS, INPUTS, OUTPUTS)

    # %% Data set-up
    # train_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=LOOKBACK, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=1, end=510+1)
    train_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=LOOKBACK, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=1, end=50)
    train_dataset.sciann = False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    val_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=LOOKBACK, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=510, end=638+1)
    val_dataset.sciann = False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    # %% CUDA set-up
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'CUDA is {is_cuda}')

    # %% Model set-up
    model = models.PINNS(len(INPUTS), 256, len(OUTPUTS))
    model.to(device)
    print(model)

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=REDUCE_LR_PATIENCE)

    # %% Training loop
    best_loss = np.inf
    best_loss_separate = []
    best_epoch = -1

    train_history = []
    val_history = []
    
    start_time = timer()
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch+1}/{N_EPOCHS}')
        # Batch Training
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc='Training') as p_bar:
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                opt.zero_grad()
                
                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                opt.step()

                # Update training batch stats
                train_loss += loss.item()
                p_bar.update()
                p_bar.postfix = f'train_loss: {train_loss/(step+1):.5f}'

        # Validation
        model.eval()
        val_loss = 0
        separate_val_mses = []
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validating') as p_bar:
                for step, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    y_pred = model(batch_x)
                    loss = loss_fn(y_pred, batch_y)

                    separate_val_mses.append(mean_squared_error(batch_y.cpu(), y_pred.cpu(), multioutput="raw_values"))

                    # Update validation batch stats
                    val_loss += loss.item()
                    p_bar.update()
                    p_bar.postfix = f'val_loss: {val_loss/(step+1):.5f}'

        # Update training stats
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        separate_val_mses = np.array(separate_val_mses).mean(axis=0)

        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f'\tEpoch {epoch+1} Stats | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | val_mses: {separate_val_mses}')
        
        # Callbacks
        if best_loss - val_loss > 0.001:
            best_loss = val_loss
            best_loss_separate = separate_val_mses
            best_epoch = epoch

            state = {
                'epoch': epoch,
                'optimizer': opt.state_dict(),
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_loss_separate': separate_val_mses
            }
            torch.save(state, f'checkpoints/LSTM_torch_exp{EXPERIMENT_N}-best.pth.tar')
        else:
            epochs_without_improving = epoch - best_epoch
            if epochs_without_improving < EARLY_STOP_PATIENCE:
                print(f'\tVal loss did not improve from {best_loss} | {epochs_without_improving} epochs without improvement')
            else:
                print(f'\tVal loss did not improve from {best_loss}, patience ran out so stopping early')
                break
    
    # %% Post-Training
    duration = timer() - start_time
    print(f'Training took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h')

    fig, ax = plt.subplots()
    ax.plot(train_history, label = 'train')
    ax.plot(val_history, label = 'val')
    ax.set_title('Loss (Mean Squared Error)')
    ax.legend(loc='upper right')
    plt.savefig(f'plots/lstm_exp_{EXPERIMENT_N}.png')

    #%% Send a text message via Twilio
    client.messages.create(
        body=f'PyTorch Model {EXPERIMENT_N} has completed with val_loss {best_loss} | individual={best_loss_separate}',
        from_=keys.src_phone,
        to=keys.dst_phone
    )