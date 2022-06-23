import os
import joblib
from pathlib import Path
from typing import List

import numpy as np
from tensorflow import keras
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from data import read_np


class __StandardScaler3D__(StandardScaler):
    def partial_fit(self, X, y=None, sample_weight=None):
        return super().partial_fit(X.reshape(X.shape[0]*X.shape[1], -1), y=y, sample_weight=sample_weight)

    def transform(self, X, copy=None):
        return super().transform(X.reshape(X.shape[0]*X.shape[1], -1), copy=copy).reshape(X.shape)

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplemented()

    def fit(self, X, y=None, sample_weight=None):
        raise NotImplemented()


def load_or_create(scaler_path: str, creation_dirs: List[str], inputs: List[str], outputs: List[str], use_2D: bool = False):
    if os.path.exists(scaler_path):
        print('Loading previous scaler')
        SCALER = joblib.load(scaler_path)
    else:
        print('Recreating scaler')
        if use_2D:
            SCALER = __StandardScaler3D__()
        else:
            SCALER = StandardScaler()

        for idx, creation_dir in enumerate(creation_dirs):
            print(f"Processing dir {idx+1}/{len(creation_dirs)}")
            for filepath in tqdm(list(Path(creation_dir).glob('*.npy'))):
                X, Y = read_np(filepath, inputs, outputs, use_2D, scaler=None)
                SCALER.partial_fit(X)
        joblib.dump(SCALER, scaler_path)
