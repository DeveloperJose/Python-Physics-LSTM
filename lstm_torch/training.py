import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
from pathlib import Path

import keys
from settings import Settings as S

def train(device, model, opt, n_epochs, train_loader, val_loader, client):
    opt_name = opt.__class__.__name__
    print('Training with', opt_name)
    
    best_loss = np.inf
    best_loss_separate = []
    best_epoch = -1

    train_history = []
    val_history = []

    checkpoint_filename = S.CHECKPOINTS_PATH / f'LSTM_torch_exp{S.EXPERIMENT_N}_{opt_name}-best.pth.tar'
    plot_filename = S.PLOTS_PATH / f'lstm_exp_{S.EXPERIMENT_N}_{opt_name}.png'
    
    start_time = timer()
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')
        # Batch Training
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc='Training') as p_bar:
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                def closure():
                    opt.zero_grad()
                    losses = model.losses(batch_x, batch_y)
                    loss = sum(losses)
                    loss.backward()
                    return loss

                opt.step(closure)
                loss = closure()

                # Update training batch stats
                train_loss += loss.item()
                p_bar.update()
                p_bar.postfix = f'train_loss: {train_loss/(step+1):.5f}'

        # Validation. I need the gradients so don't call torch.no_grad()!
        model.eval()
        val_loss = 0
        separate_val_mses = []

        with tqdm(total=len(val_loader), desc='Validating') as p_bar:
            for step, (batch_x, batch_y) in enumerate(val_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                losses = model.losses(batch_x, batch_y)
                loss = sum(losses)

                # separate_val_mses.append(mean_squared_error(batch_y.detach().cpu(), y_pred.detach().cpu(), multioutput="raw_values"))
                separate_val_mses.append([l.cpu().item() for l in losses])

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
        output_str = f'\tEpoch {epoch+1} Stats | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | val_mses: {separate_val_mses}'
        if model.use_pinns:
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

            state = {
                'epoch': epoch,
                'optimizer': opt.state_dict(),
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_loss_separate': best_loss_separate
            }
            torch.save(state, checkpoint_filename)
        else:
            epochs_without_improving = epoch - best_epoch
            if epochs_without_improving < S.EARLY_STOP_PATIENCE:
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
    plt.savefig(plot_filename)

    #%% Send a text message via Twilio
    client.messages.create(
        body=f'PyTorch Model {S.EXPERIMENT_N} for optimizer {opt_name} has completed with val_loss {best_loss} | individual={best_loss_separate}',
        from_=keys.src_phone,
        to=keys.dst_phone
    )