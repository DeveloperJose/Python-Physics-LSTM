import logging
import os
from pathlib import Path
from typing import List

import data
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger("physics_lstm")


def load_or_create(scaler_path: str, creation_dirs: List[str], inputs: List[str], outputs: List[str]):
    if os.path.exists(scaler_path):
        logger.info("Loading previous scaler")
        SCALER = joblib.load(scaler_path)
    else:
        logger.info("Recreating scaler")
        SCALER = StandardScaler()

        for idx, creation_dir in enumerate(creation_dirs):
            logger.info(f"Processing dir {idx+1}/{len(creation_dirs)}")
            for filepath in tqdm(list(Path(creation_dir).glob("*.npy"))):
                X, Y = data.read_np(filepath, inputs, outputs, scaler=None)
                SCALER.partial_fit(X)
        joblib.dump(SCALER, scaler_path)
