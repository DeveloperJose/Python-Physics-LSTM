import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
input_path = Path('/home/jperez/data/sled250/')
output_dir = Path('/home/jperez/data/sled250_2D/')

SCALE_2D = 1000
X_MAX = 4 * SCALE_2D
Y_MAX = 2.2 * SCALE_2D
STEP_2D = 0.025 * SCALE_2D
IM_ROWS = int(Y_MAX/STEP_2D)
IM_COLS = int(X_MAX/STEP_2D)
ROW_F = int(IM_ROWS/Y_MAX*SCALE_2D)
COL_F = int(IM_COLS/X_MAX*SCALE_2D)

def read_np_2d(filename, scaler=None):
    arr = np.load(filename).astype(np.float64)
    
    # input_idxs = [COLUMNS_H.index(s) for s in inputs_str]
    # output_idxs = [COLUMNS_H.index(s) for s in outputs_str]
    # X_IN = arr[:, input_idxs]
    # Y_OUT = arr[:, output_idxs]

    im = np.zeros((IM_ROWS+1, IM_COLS+1), dtype=np.uint8)
    arr[:, 0] += 0.025
    arr[:, 0] *= SCALE_2D
    arr[:, 1] *= SCALE_2D

    for x, y, p, vu, vv in zip(arr[:, 0], arr[:, 1], arr[:, 3], arr[:, 4], arr[:, 5]):
        r = round((y * ROW_F)/SCALE_2D)
        c = round((x * COL_F)/SCALE_2D)
        #im[r, c] = [vu, 0, 0]
        im[r, c] = vu

    if scaler:
        im = scaler.transform(im)

    return im[::-1]

# scaler = StandardScaler()
print(f'Converting {input_path} to 2D, storing results in {output_dir}')
for filepath in tqdm(list(input_path.glob('*.npy'))):
    im = read_np_2d(filepath)
    # scaler.partial_fit(im)
    np.save(output_dir / f'{filepath.name}', im)

# print(f'Scaler info: mean={scaler.mean_}, scale={scaler.scale_}, var={scaler.var_}')

# print(f'Saving scaler')
# joblib.dump(scaler, 'scaler_2d.pkl')