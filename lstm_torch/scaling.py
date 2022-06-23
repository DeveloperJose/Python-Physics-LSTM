import os
import joblib
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import data

def load_or_create(scaler_path: str, creation_dirs: List[str], inputs: List[str], outputs: List[str]):
    if os.path.exists(scaler_path):
        print('Loading previous scaler')
        SCALER = joblib.load(scaler_path)
    else:
        print('Recreating scaler')
        SCALER = StandardScaler()

        for idx, creation_dir in enumerate(creation_dirs):
            print(f"Processing dir {idx+1}/{len(creation_dirs)}")
            for filepath in tqdm(list(Path(creation_dir).glob('*.npy'))):
                X, Y = data.read_np(filepath, inputs, outputs, scaler=None)
                SCALER.partial_fit(X)
        joblib.dump(SCALER, scaler_path)

# def load_or_create_2d(scaler_path, creation_dirs):
#     if os.path.exists(scaler_path):
#         print('Loading previous 2D scaler')
#         SCALER = joblib.load(scaler_path)
#     else:
#         print('Recreating 2D scaler')
#         SCALER = StandardScaler()

#         for idx, creation_dir in enumerate(creation_dirs):
#             print(f"Processing dir {idx+1}/{len(creation_dirs)}")
#             for filepath in tqdm(list(Path(creation_dir).glob('*.npy'))):
#                 im = data.read_np_2d(filepath)
#                 SCALER.partial_fit(im)
#         joblib.dump(SCALER, scaler_path)