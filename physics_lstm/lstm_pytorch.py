import logging
from pprint import pformat
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from twilio.rest import Client

import physics_lstm.data as data
import physics_lstm.keys as keys
import physics_lstm.models as models
import physics_lstm.scaling as scaling
from physics_lstm.configuration import Configuration

np.set_printoptions(suppress=True, linewidth=np.inf)  # No more scientific notation
logger = logging.getLogger("physics_lstm")


def save_model(filename, model, opt, val_loss, epoch, settings):
    state = {
        "epoch": epoch,
        "optimizer": opt.state_dict(),
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "settings": settings,
    }
    torch.save(state, filename)


def setup_cuda(model):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"CUDA is {is_cuda} and {torch.cuda.device_count()} GPUs")
    model.to(device)
    return device


def step(model, opt, device, data_loader, is_training=True):
    """Perform a single step of training or validation.
    Args:
        model: The model to train or validate
        opt: The optimizer to use
        device: The device to use
        data_loader: The data loader to use
        is_training: Whether to train or validate
    """
    if is_training:
        desc = "Training"
        loss_name = "train_loss"
    else:
        desc = "Validating"
        loss_name = "val_loss"

    step_loss = 0
    separate_losses = []

    with tqdm(total=len(data_loader), desc=desc) as p_bar:
        for step, (batch_x, batch_y) in enumerate(data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if is_training:
                opt.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                losses = model.losses(batch_x, batch_y)
                loss = sum(losses)
                loss.backward()
                opt.step()
            else:
                # For validation I need the gradients so don't call torch.no_grad() even though it speeds up training!
                losses = model.losses(batch_x, batch_y)
                loss = sum(losses)
                separate_losses.append([ls.cpu().item() for ls in losses])

            # Update batch stats
            step_loss += loss.item()
            p_bar.update()
            p_bar.postfix = f"{loss_name}: {step_loss/(step+1):.5f}"

    return step_loss / len(data_loader), separate_losses


def plot_history(train_history, val_history, plot_filename):
    fig, ax = plt.subplots()
    ax.plot(train_history, label="train")
    ax.plot(val_history, label="val")
    ax.set_title("Loss (Mean Squared Error)")
    ax.legend(loc="upper right")
    plt.savefig(plot_filename)

def main():
    # %% Config set-up
    config = Configuration.load_from_yaml(Path("config/paper.yaml"))

    # %% Logger set-up
    logger_filepath = Path("output") / f"exp_{config.experiment_name}.log"
    # if logger_filepath.exists():
    #     raise Exception("Experiment with the same name already has existing log file")
    logging.basicConfig(level=logging.INFO, filename=logger_filepath)
    logging.info("Configuration")
    logging.info(pformat(config))

    # %% Twilio set-up
    if config.use_twilio:
        client = Client(keys.account_sid, keys.auth_token)

    # %% Scaler set-up
    scaler = scaling.load_or_create(config.scaler_path, config.scaler_creation_dirs, config.inputs, config.outputs)

    # %% Data set-up
    # batch_size = config.second_batch_size if config.use_pinns else config.first_batch_size
    batch_size = config.batch_size

    # sled250 (1-742) | sled300 (1-742) | sled350 (1-742)
    train_dataset = data.SledDataGenerator(
        config.train_data_path,
        sequence_length=config.seq_len,
        inputs=config.inputs,
        outputs=config.outputs,
        scaler=scaler,
        dropin=config.train_dropin,
        start=config.train_start_timestep,
        end=config.train_end_timestep,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.n_workers, pin_memory=True
    )

    val_dataset = data.SledDataGenerator(
        config.val_data_path,
        sequence_length=config.seq_len,
        inputs=config.inputs,
        outputs=config.outputs,
        scaler=scaler,
        dropin=config.val_dropin,
        start=config.val_start_timestep,
        end=config.val_end_timestep,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=config.n_workers, pin_memory=True
    )

    # %% Baseline
    if config.use_baseline:
        logger.info("Running baseline")
        y_true = []
        y_pred = []

        start_time = timer()
        for timestep in range(1, val_dataset.x_data.shape[0] - config.seq_len):
            # Get the whole lookup slice
            prev_data = val_dataset.x_data[timestep : timestep + config.seq_len]
            # Only use the last (u,v) for the no_change baseline
            prev_uv = prev_data[-1, :, [3, 4]]
            y_pred.extend(np.array(prev_uv))

            # Get the next timestep after the lookup
            next_data = val_dataset.x_data[timestep + config.seq_len]
            next_uv = next_data[:, [3, 4]]
            y_true.extend(np.array(next_uv))

        duration = timer() - start_time
        baseline_mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
        logger.info(f"MSE={baseline_mse}, took {duration:.3f}sec")
        return

    # %% Model set-up
    model = models.LSTM_PINNS(config)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=config.reduce_lr_patience)
    logger.info(f"{model}")

    # %% CUDA set-up
    device = setup_cuda(model)

    # %% Training loop
    opt_name = opt.__class__.__name__
    logger.info(f"Training with {opt_name}")

    start_epoch = 0
    best_loss = np.inf
    best_loss_separate = []
    best_epoch = -1

    train_history = []
    val_history = []

    if config.prev_checkpoint.exists() and config.prev_checkpoint.is_file():
        logger.info("Loading previous checkpoint", config.prev_checkpoint)
        checkpoint = torch.load(config.prev_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]

    checkpoint_final_filepath = config.checkpoints_path / f"exp{config.experiment_name}-best.pth.tar"
    plot_final_filepath = config.plots_path / f"exp_{config.experiment_name}-final.png"

    # %% Check if we forgot to update the experiment number
    if checkpoint_final_filepath.exists():
        raise Exception("Experiment with the same name already has existing checkpoint")

    start_time = timer()
    for epoch in range(start_epoch, config.n_epochs + start_epoch):
        logger.info(f"Epoch {epoch+1}/{config.n_epochs+start_epoch}")

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

        output_str = f"\tEpoch {epoch+1} Stats | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f} | val_mses: {separate_val_mses}"
        if config.use_pinns:
            lambda1 = model.lambda1.detach().cpu().item()
            lambda2 = model.lambda2.detach().cpu().item()
            lstm_w = model.lstm_w.detach().cpu().item()
            lstm_w_sigmoid = torch.sigmoid(model.lstm_w.detach().cpu()).item()
            output_str += f" | lambda1: {lambda1:.10f} | lambda2: {lambda2:.10f} | lstm_w: {lstm_w:.3f} -> sigmoid: {lstm_w_sigmoid:.3f}"
        logger.info(output_str)

        # Callbacks
        if best_loss - val_loss > 0.001:
            best_loss = val_loss
            best_loss_separate = separate_val_mses
            best_epoch = epoch
            save_model(checkpoint_final_filepath, model, opt, val_loss, epoch, config)
        else:
            epochs_without_improving = epoch - best_epoch
            if epochs_without_improving < config.early_stop_patience:
                logger.info(
                    f"\tVal loss did not improve from {best_loss} | {epochs_without_improving} epochs without improvement"
                )
            else:
                logger.info(f"\tVal loss did not improve from {best_loss}, patience ran out so stopping early")
                break

    # %% Post-Training
    duration = timer() - start_time
    logger.info(f"\tTraining took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h")
    logger.info(f"\tThe best val_loss was {best_loss} at epoch {best_epoch} with MSEs {best_loss_separate}")

    # %% Plot history
    plot_history(train_history, val_history, plot_final_filepath)

    # %% Send a text message via Twilio
    if config.use_twilio:
        client.messages.create(
            body=f"PyTorch Model {config.experiment_name} for optimizer {opt_name} has completed with best val_loss {best_loss} in epoch {best_epoch} after {epoch+1} epochs | individual={best_loss_separate}",
            from_=keys.src_phone,
            to=keys.dst_phone,
        )


if __name__ == "__main__":
    main()
