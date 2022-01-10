import os
from pathlib import Path
from typing import List

import numpy as np
from tensorflow import keras
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

WIDTH = 197
HEIGHT = 72
COLUMNS_H = ['X', 'Y', 'P', 'Vu', 'Vv', 'W.VF']
COLUMNS_RAW = ['X [ m ]', 'Y [ m ]', 'Pressure [ Pa ]',
               'Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]', 'Water.Volume Fraction']


def read_np(filename, inputs_str, outputs_str, use_2D=False, scaler=None):
    arr = np.load(filename)

    input_idxs = [COLUMNS_H.index(s) for s in inputs_str]
    output_idxs = [COLUMNS_H.index(s) for s in outputs_str]
    X = arr[:, input_idxs]
    Y = arr[:, output_idxs]

    # Convert pressure from Pa to Atm
    p_idx = inputs_str.index('P')
    X[:, p_idx] = X[:, p_idx] / 101325

    if scaler:
        X = scaler.transform(X)

    if use_2D:
        X = X.reshape((HEIGHT, WIDTH, len(input_idxs)))
        Y = Y.reshape((HEIGHT, WIDTH, len(output_idxs)))

    return X, Y


# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class SledDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir: str, batch_size: int, lookback: int, shuffle: bool, use_2D: bool, inputs: List[str], outputs: List[str], scaler: StandardScaler, start: int, end: int, step: int = 1):
        print(
            f'Loading dataset {data_dir} from t={start} to t={end} with 2D={use_2D}')

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle
        self.use_2D = use_2D
        self.inputs = inputs
        self.outputs = outputs
        self.scaler = scaler
        self.start = start
        self.end = end
        self.step = step

        # Check if we have a serialized version of the data, if not, generate it
        data_path = os.path.join(
            'output', f'{self.data_dir.stem}_{start}_{end}_in_{inputs}_out_{outputs}_2d_{use_2D}.npz')
        if os.path.exists(data_path):
            csv_data = np.load(data_path)
            self.x_data = csv_data['X']
            self.y_data = csv_data['Y']
        else:
            self.x_data = []
            self.y_data = []

            for timestep in tqdm(range(start, end, step)):
                X, Y = self.__read_np__(timestep)
                self.x_data.append(X)
                self.y_data.append(Y)
            self.x_data = np.array(self.x_data)
            self.y_data = np.array(self.y_data)
            np.savez(data_path, X=self.x_data, Y=self.y_data)
        print('Debug: X=', self.x_data.shape, 'Y=', self.y_data.shape)

        # Some sanity checks
        if not use_2D:
            assert self.x_data.shape[1] == self.y_data.shape[1], 'x_data and y_data have a shape mismatch in the number of CSV rows'

        # Helpful variables
        # 1D: [timestep, inputs, 1]
        # 2D: [timestep, height, width, inputs, 1]
        self.n_timesteps = self.x_data.shape[0]

        # Generate a list of the valid timesteps for batches
        self.list_timesteps = np.arange(lookback, self.n_timesteps)
        self.list_rows = np.arange(self.x_data.shape[1])
        self.list_IDs = [(t, r)
                         for t in self.list_timesteps for r in self.list_rows]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = self.indexes[index *
                                  self.batch_size:(index+1)*self.batch_size]
        batch_pairs = [self.list_IDs[k] for k in batch_idxs]

        batch_x = []
        batch_y = []
        for (timestep, row) in batch_pairs:
            if self.use_2D:
                X = self.x_data[timestep-self.lookback:timestep, :, :]
                Y = self.y_data[timestep, :, :]
            else:
                X = self.x_data[timestep-self.lookback:timestep, row]
                Y = self.y_data[timestep, row]

            batch_x.append(X)
            batch_y.append(Y)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __read_np__(self, timestep):
        filename = os.path.join(self.data_dir, f'{timestep}.npy')
        return read_np(filename, self.inputs, self.outputs, self.use_2D, self.scaler)
