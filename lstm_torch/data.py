import os
from typing import List
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

import models
# WIDTH = 197
# HEIGHT = 72
COLUMNS_H = ['X', 'Y', 'T', 'P', 'Vu', 'Vv', 'W.VF']
COLUMNS_RAW = ['X [ m ]', 'Y [ m ]', 'Timestep', 'Pressure [ Pa ]',
               'Velocity u [ m s^-1 ]', 'Velocity v [ m s^-1 ]', 'Water.Volume Fraction']


def read_np(filename, inputs_str, outputs_str, scaler=None):
    arr = np.load(filename)

    input_idxs = [COLUMNS_H.index(s) for s in inputs_str]
    output_idxs = [COLUMNS_H.index(s) for s in outputs_str]
    X = arr[:, input_idxs]
    Y = arr[:, output_idxs]

    # Convert pressure from Pa to Atm
    try:
        p_idx = inputs_str.index('P')
        X[:, p_idx] = X[:, p_idx] / 101325
    except ValueError:
        pass

    if scaler:
        X = scaler.transform(X)

    # if use_2D:
    #     X = X.reshape((HEIGHT, WIDTH, len(input_idxs)))
    #     Y = Y.reshape((HEIGHT, WIDTH, len(output_idxs)))

    return X, Y


class SledDataGenerator(Dataset):
    def __init__(self, data_dir, sequence_length, inputs, outputs, scaler, start, end, step = 1):
        print(f'Loading dataset {data_dir} from t={start} to t={end}')

        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length

        self.inputs = inputs
        self.outputs = outputs
        self.scaler = scaler

        self.start = start
        self.end = end
        self.step = step

        # Hidden states
        self.sciann = False

        # Check if we have a serialized version of the data, if not, generate it
        data_path = os.path.join('output', f'{self.data_dir.stem}_{start}_{end}_in_{inputs}_out_{outputs}.npz')
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

        self.x_data = torch.tensor(self.x_data).float()
        self.y_data = torch.tensor(self.y_data).float()

        # Generate a list of the valid timesteps
        self.n_timesteps = self.x_data.shape[0]
        # self.list_timesteps = np.arange(self.sequence_length, self.n_timesteps)
        self.list_timesteps = np.arange(self.n_timesteps)
        self.list_rows = np.arange(self.x_data.shape[1])
        self.list_IDs = [(t, r) for t in self.list_timesteps for r in self.list_rows]

    def __len__(self):
        return len(self.list_IDs)

    # https://www.crosstab.io/articles/time-series-pytorch-lstm
    def __getitem__(self, index):
        timestep, row = self.list_IDs[index]
        # Check if we are at the beginning and need to pad
        if timestep >= self.sequence_length-1:
            # This one doesn't include current
            # X = self.x_data[timestep-self.sequence_length:timestep, row]

            X = self.x_data[timestep-self.sequence_length+1:timestep+1, row]
        else:
            padding = self.x_data[0, row].repeat(self.sequence_length-timestep-1, 1)
            X = self.x_data[0:(timestep+1), row]
            X = torch.cat((padding, X), 0)

        Y = self.y_data[timestep, row]

        # if self.model_state == models.State.BOTH_BRANCHES or self.model_state == models.State.PINNS_ONLY:
        #     Y = torch.cat((Y, torch.zeros(2)))

        # # Check if we are at the beginning and need to pad
        # if timestep >= self.sequence_length - 1:
        #     start_idx = timestep - self.sequence_length + 1
        #     X = self.x_data[start_idx:(timestep+1), :]
        # else:
        #     padding = self.x_data[0:1].repeat(self.sequence_length-timestep-1, 1, 1)
        #     X = self.x_data[0:(timestep+1), :]
        #     X = torch.cat((padding, X), 0)
        # Y = self.y_data[timestep, :]

        # if self.sciann:
            # Y = np.append(Y, [0.0, 0.0])
        
        return X, Y

    def __read_np__(self, timestep):
        filename = os.path.join(self.data_dir, f'{timestep}.npy')
        return read_np(filename, self.inputs, self.outputs, self.scaler)
