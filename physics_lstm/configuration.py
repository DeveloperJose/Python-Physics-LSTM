from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml


@dataclass
class Configuration:
    experiment_name: str
    inputs: List[str]
    outputs: List[str]
    n_lstm_output: int
    n_dense_output: int
    use_twilio: bool

    # training parameters
    n_epochs: int
    seq_len: int
    first_batch_size: int
    second_batch_size: int
    n_workers: int
    early_stop_patience: int
    reduce_lr_patience: int
    prev_checkpoint: Path

    # training data
    train_data_path: Path
    train_dropin: float
    train_start_timestep: int
    train_end_timestep: int

    val_data_path: Path
    val_dropin: float
    val_start_timestep: int
    val_end_timestep: int

    # outputs
    plots_path: Path
    checkpoints_path: Path
    scaler_path: Path
    scaler_creation_dirs: List[Path]

    # network architecture
    use_lstm: bool
    use_pinns: bool

    bidirectional_lstm: bool
    n_lstm_layers: int
    lstm_activations: int
    lstm_td_activations: int
    lstm_n_dense_layers: int
    lstm_dense_activations: int

    n_dense_layers: int
    dense_activations: int

    def __post_init__(self):
        if self.reduce_lr_patience > self.early_stop_patience:
            raise Exception("Training will stop early before reducing LR")

        self.train_data_path = Path(self.train_data_path)
        self.val_data_path = Path(self.val_data_path)

        self.plots_path = Path(self.plots_path)
        self.checkpoints_path = Path(self.checkpoints_path)
        self.scaler_path = Path(self.scaler_path)
        self.scaler_creation_dirs = [Path(d) for d in self.scaler_creation_dirs]
        self.prev_checkpoint = Path(self.prev_checkpoint)

        # Calculated
        self.n_inputs = len(self.inputs)
        self.n_outputs = len(self.outputs)

    @staticmethod
    def load_from_yaml(filename: Union[str, Path]):
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist or the wrong path was provided")

        with open(filename, "r") as file:
            parsed_dict = yaml.safe_load(file)
            return Configuration(**parsed_dict)


if __name__ == "__main__":
    settings = Configuration.load_from_yaml("config/paper.yaml")
