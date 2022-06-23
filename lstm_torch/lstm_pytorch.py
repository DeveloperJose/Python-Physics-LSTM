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
np.set_printoptions(suppress=True, linewidth=np.inf) # No more scientific notation

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Other libraries
from twilio.rest import Client
from tqdm import tqdm

# Project libraries
# import training
import scaling
import data
import keys
import models

class Settings:
    EXPERIMENT_N = 43

    INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
    OUTPUTS = ['Vu', 'Vv']

    # USE_VU = 'Vu' in OUTPUTS
    # USE_VV = 'Vv' in OUTPUTS

    # Training Parameters
    N_EPOCHS = 100
    SEQ_LEN = 3 # AKA: Lookback
    FIRST_BATCH_SIZE = 20000 # 25000 # 200000
    SECOND_BATCH_SIZE = 5000 # 20000
    N_WORKERS = 6
    EARLY_STOP_PATIENCE = 50 # 50
    REDUCE_LR_PATIENCE = 25 # 25

    # Data Loading
    PLOTS_PATH = Path('plots')
    CHECKPOINTS_PATH = Path('checkpoints')
    SCALER_PATH = os.path.join('output', f'scalerv3_{INPUTS}.pkl')
    SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled300']

    PREV_CHECKPOINT = None # Path('checkpoints') / 'LSTM_torch_exp30_Adam-final-best.pth.tar' # Path('best') / 'model.pth.tar'

    # Network Architecture
    USE_LSTM = True
    USE_PINNS = True

    BIDIRECTIONAL_LSTM = True
    N_LSTM_LAYERS = 2
    LSTM_ACTIVATIONS = 32
    LSTM_TD_ACTIVATIONS = 32
    LSTM_N_DENSE_LAYERS = 1
    LSTM_DENSE_ACTIVATIONS = 32

    N_DENSE_LAYERS = 1
    DENSE_ACTIVATIONS = 32

    # Constants
    N_INPUTS = len(INPUTS)
    N_LSTM_OUTPUT = len(OUTPUTS) # [Vu, Vv]
    N_DENSE_OUTPUT = 2 # [Psi, P]

S = Settings

def save_model(model, opt, val_loss, epoch, filename):
    state = {
        'epoch': epoch,
        'optimizer': opt.state_dict(),
        'state_dict': model.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, filename)

def setup_cuda(model):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        # model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
    print(f'CUDA is {is_cuda} and {torch.cuda.device_count()} GPUs')
    model.to(device)
    return device

def step(model, opt, device, data_loader, is_training=True):
    if is_training:
        desc = 'Training'
        loss_name = 'train_loss'
    else:
        desc = 'Validating'
        loss_name = 'val_loss'

    step_loss = 0
    separate_losses = []
    with tqdm(total=len(data_loader), desc=desc) as p_bar:
        for step, (batch_x, batch_y) in enumerate(data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if is_training:
                opt.zero_grad(set_to_none=True) # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                losses = model.losses(batch_x, batch_y)
                loss = sum(losses)
                loss.backward()
                opt.step()
            else:
                # For validation I need the gradients so don't call torch.no_grad() even though it speeds up training!
                losses = model.losses(batch_x, batch_y)
                loss = sum(losses)
                separate_losses.append([l.cpu().item() for l in losses])

            # Update batch stats
            step_loss += loss.item()
            p_bar.update()
            p_bar.postfix = f'{loss_name}: {step_loss/(step+1):.5f}'

    return step_loss / len(data_loader), separate_losses

def plot_history(train_history, val_history, plot_filename):
    fig, ax = plt.subplots()
    ax.plot(train_history, label = 'train')
    ax.plot(val_history, label = 'val')
    ax.set_title('Loss (Mean Squared Error)')
    ax.legend(loc='upper right')
    plt.savefig(plot_filename)

if __name__ == '__main__':
    assert S.REDUCE_LR_PATIENCE < S.EARLY_STOP_PATIENCE, 'Training will stop early before reducing LR'

    # %% Twilio set-up
    client = Client(keys.account_sid, keys.auth_token)

    # %% Scaler set-up
    scaler = scaling.load_or_create(S.SCALER_PATH, S.SCALER_CREATION_DIRS, S.INPUTS, S.OUTPUTS)
    
    # %% Data set-up
    batch_size = S.SECOND_BATCH_SIZE if S.USE_PINNS else S.FIRST_BATCH_SIZE

    # sled250 (1-742) | sled300 (1-742) | sled350 (1-742)
    train_dataset = data.SledDataGenerator('/home/jperez/data/sled250', sequence_length=S.SEQ_LEN, inputs=S.INPUTS, outputs=S.OUTPUTS, scaler=scaler, 
                                            dropin=0.5, start=1, end=742)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=S.N_WORKERS, pin_memory=True)

    val_dataset = data.SledDataGenerator('/home/jperez/data/sled300', sequence_length=S.SEQ_LEN, inputs=S.INPUTS, outputs=S.OUTPUTS, scaler=scaler, 
                                            dropin=0, start=1, end=742)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=S.N_WORKERS, pin_memory=True)
    
    # %% Model set-up
    model = models.LSTM_PINNS(S)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=S.REDUCE_LR_PATIENCE)
    print(f'{model}')

    # %% CUDA set-up
    device = setup_cuda(model)

    # %% Training loop
    opt_name = opt.__class__.__name__
    print('Training with', opt_name)
    
    start_epoch = 0
    best_loss = np.inf
    best_loss_separate = []
    best_epoch = -1

    train_history = []
    val_history = []

    if S.PREV_CHECKPOINT:
        print('Loading previous checkpoint', S.PREV_CHECKPOINT)
        checkpoint = torch.load(S.PREV_CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # checkpoint_mid_filename = S.CHECKPOINTS_PATH / f'LSTM_torch_exp{S.EXPERIMENT_N}_{opt_name}-mid-best.pth.tar'
    checkpoint_final_filename = S.CHECKPOINTS_PATH / f'LSTM_torch_exp{S.EXPERIMENT_N}_{opt_name}-final-best.pth.tar'
    # plot_mid_filename = S.PLOTS_PATH / f'lstm_exp_{S.EXPERIMENT_N}_{opt_name}_mid.png'
    plot_final_filename = S.PLOTS_PATH / f'lstm_exp_{S.EXPERIMENT_N}_{opt_name}_final.png'

    # %% Check if we forgot to update the experiment number
    assert not os.path.exists(checkpoint_final_filename), 'Experiment number already exists'
    
    # Force LSTM_Only for half of the epochs
    # print('Disabling PINNs for pre-training')
    # S.USE_LSTM = True
    # S.USE_PINNS = False

    start_time = timer()
    for epoch in range(start_epoch, S.N_EPOCHS+start_epoch):
        print(f'Epoch {epoch+1}/{S.N_EPOCHS+start_epoch}')

        # Enable PINNs halfway through epochs
        # if epoch+1 == S.N_EPOCHS//2:
        # if epoch+1 == 2:

        # Enable PINNs when validation loss is below 5
        # if best_loss <= 5:
        #     print(f'Enabling PINNs at epoch {epoch} and saving mid progress')
        #     torch.cuda.empty_cache()
        #     S.USE_PINNS = True
        #     train_loader = DataLoader(train_dataset, batch_size=S.SECOND_BATCH_SIZE, shuffle=True, num_workers=S.N_WORKERS, pin_memory=True)
        #     val_loader = DataLoader(val_dataset, batch_size=S.SECOND_BATCH_SIZE, shuffle=True, num_workers=S.N_WORKERS, pin_memory=True)
        #     save_model(model, opt, best_loss, epoch, checkpoint_mid_filename)
        #     plot_history(train_history, val_history, plot_mid_filename)
        
        # Batch Training
        model.train()
        train_loss, _ = step(model, opt, device, train_loader, is_training=True)
        train_history.append(train_loss)

        # Epoch Validation
        model.eval()
        val_loss, separate_val_mses = step(model, opt, device, val_loader, is_training=False)
        scheduler.step(val_loss)
        val_history.append(val_loss)
        separate_val_mses = np.array(separate_val_mses).mean(axis=0)

        output_str = f'\tEpoch {epoch+1} Stats | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | val_mses: {separate_val_mses}'
        if S.USE_PINNS:
            lambda1 = model.lambda1.detach().cpu().item()
            lambda2 = model.lambda2.detach().cpu().item()
            lstm_w = model.lstm_w.detach().cpu().item()
            lstm_w_sigmoid = torch.sigmoid(model.lstm_w.detach().cpu()).item()
            output_str += f' | lambda1: {lambda1:.10f} | lambda2: {lambda2:.10f} | lstm_w: {lstm_w:.3f} -> sigmoid: {lstm_w_sigmoid:.3f}'
        print(output_str)
        
        # Callbacks
        if best_loss - val_loss > 0.001:
            best_loss = val_loss
            best_loss_separate = separate_val_mses
            best_epoch = epoch
            save_model(model, opt, val_loss, epoch, checkpoint_final_filename)
        else:
            epochs_without_improving = epoch - best_epoch
            if epochs_without_improving < S.EARLY_STOP_PATIENCE:
                print(f'\tVal loss did not improve from {best_loss} | {epochs_without_improving} epochs without improvement')
            else:
                print(f'\tVal loss did not improve from {best_loss}, patience ran out so stopping early')
                break
    
    # %% Post-Training
    duration = timer() - start_time
    print(f'\tTraining took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h')
    print(f'\tThe best val_loss was {best_loss} at epoch {best_epoch} with MSEs {best_loss_separate}')

    # %% Plot history
    plot_history(train_history, val_history, plot_final_filename)

    #%% Send a text message via Twilio
    client.messages.create(
        body=f'PyTorch Model {S.EXPERIMENT_N} for optimizer {opt_name} has completed with best val_loss {best_loss} in epoch {best_epoch} after {epoch+1} epochs | individual={best_loss_separate}',
        from_=keys.src_phone,
        to=keys.dst_phone
    )