from pathlib import Path
from numba import jit

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow.keras as keras

SCALE_2D = 1000
X_MAX = 4 * SCALE_2D
Y_MAX = 2.2 * SCALE_2D
STEP_2D = 0.025 * SCALE_2D
IM_ROWS = int(Y_MAX/STEP_2D)
IM_COLS = int(X_MAX/STEP_2D)
ROW_F = int(IM_ROWS/Y_MAX*SCALE_2D)
COL_F = int(IM_COLS/X_MAX*SCALE_2D)

np.set_printoptions(suppress=True, linewidth=np.inf)

@jit(nopython=True)
def dropout_fill(im: np.ndarray, binomial: np.ndarray):
    if binomial is not None:
        # Bibek: Take median multiple times. Only works well when dropout is small
        for r in np.arange(1, im.shape[0]):
            for c in np.arange(1, im.shape[1]):
                im[r, c] = np.sum(im[r-1:r+2, c-1:c+2]) / np.sum(im[r-1:r+2, c-1:c+2] > 0)
    return im

@jit(nopython=True)
def read_np_2d(arr, binomial):
    im = np.zeros((IM_ROWS+1, IM_COLS+1, 2), dtype=np.uint8)
    arr[:, 0] += 0.025
    arr[:, 0] *= SCALE_2D
    arr[:, 1] *= SCALE_2D

    # for x, y, p, vu, vv in zip(arr[:, 0], arr[:, 1], arr[:, 3], arr[:, 4], arr[:, 5]):
    for x, y, vu, vv in zip(arr[:, 0], arr[:, 1], arr[:, 4], arr[:, 5]):
        r = round((y * ROW_F)/SCALE_2D)
        c = round((x * COL_F)/SCALE_2D)
        im[r, c, 0] = vu
        im[r, c, 1] = vv
        if binomial is not None:
            im[r, c, 0] *= binomial[r, c]
            im[r, c, 1] *= binomial[r, c]

    return im[::-1]

class Sled2DGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, start, end, batch_size, shuffle, drop_in, predict_ahead):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.list_IDs = np.arange(start, end)
        self.on_epoch_end()

        # ***Input Dropping
        # Use file ID as random number seed so the same inputs are always dropped
        self.drop_in = drop_in
        self.predict_ahead = predict_ahead
        seed = 1738

        if drop_in > 0:
            self.rng = np.random.default_rng(seed)
            self.binomial = self.rng.binomial(n=1, p=1-drop_in, size=(IM_ROWS+1, IM_COLS+1))
            self.binomial[:, 0] = 1
            self.binomial[0, :] = 1
            print(f'{data_dir} dropping {drop_in}')
            plt.figure()
            plt.suptitle(f'Mask for seed {seed} | [{np.min(self.binomial)}, {np.max(self.binomial)}] | Drop {drop_in}')
            plt.imshow(self.binomial, cmap='gray')
            plt.savefig('mask.png')
        else:
            self.binomial = None

        # ***Input Scaling
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        for id in tqdm(self.list_IDs, desc='Creating scaler'):
            im = read_np_2d(np.load(self.data_dir / f'{id}.npy').astype(np.float64), self.binomial)
            self.scaler1.partial_fit(im[:, :, 0])
            self.scaler2.partial_fit(im[:, :, 1])

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = np.zeros((self.batch_size, IM_ROWS+1, IM_COLS+1, 2), dtype=np.float64)
        if self.predict_ahead:
            Y = np.zeros((self.batch_size, self.predict_ahead, IM_ROWS+1, IM_COLS+1, 2), dtype=np.float64)
        else:
            Y = np.zeros((self.batch_size, IM_ROWS+1, IM_COLS+1, 2), dtype=np.float64)

        for idx, id in enumerate(list_IDs_temp):
            # im = np.load(self.data_dir / f'{id}.npy')
            im = read_np_2d(np.load(self.data_dir / f'{id}.npy').astype(np.float64), self.binomial)
            im = dropout_fill(im, self.binomial)
            X[idx, :, :, 0] = self.scaler1.transform(im[:, :, 0]).reshape(89, 161)
            X[idx, :, :, 1] = self.scaler2.transform(im[:, :, 1]).reshape(89, 161)

            # np.load(self.data_dir / f'{id+1}.npy')
            if self.predict_ahead:
                for offset in range(self.predict_ahead):
                    future_fpath = self.data_dir / f'{id+offset+1}.npy'
                    if not future_fpath.exists():
                        future_fpath = self.data_dir / f'{id}.npy'
                    
                    future_im = read_np_2d(np.load(future_fpath).astype(np.float64), None)
                    Y[idx, offset, :, :, 0] = self.scaler1.transform(future_im[:, :, 0]).reshape(89, 161)
                    Y[idx, offset, :, :, 1] = self.scaler2.transform(future_im[:, :, 1]).reshape(89, 161)
            else:
                future_im = read_np_2d(np.load(self.data_dir / f'{id+1}.npy').astype(np.float64), None)
                Y[idx, :, :, 0] = self.scaler1.transform(future_im[:,:,0]).reshape(89, 161)
                Y[idx, :, :, 1] = self.scaler2.transform(future_im[:,:,1]).reshape(89, 161)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# sled250 = 1 to 741
print('Loading data')
n_ahead = 15
train_data = Sled2DGenerator('/home/jperez/data/sled250', 1, 741, batch_size=16, shuffle=True, drop_in=0.8, predict_ahead=n_ahead)
val_data = Sled2DGenerator('/home/jperez/data/sled300', 1, 741, batch_size=16, shuffle=True, drop_in=0, predict_ahead=n_ahead)

print('Preparing model')
model = keras.models.Sequential()

# SimpleCNN
# model.add(keras.layers.Conv2D(1, (3, 3), padding='same', input_shape=(89, 161, 1)))

# CNN ver 3
n_outputs = 2 # vu, vv
model.add(keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(89, 161, 2)))
model.add(keras.layers.Conv2D(8, (3, 3), padding='same'))
model.add(keras.layers.Conv2D(4, (3, 3), padding='same'))
model.add(keras.layers.Conv2D(n_outputs * n_ahead, (3, 3), padding='same'))
model.add(keras.layers.Reshape((n_ahead, 89, 161, 2)))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
model.summary()

early_stop = keras.callbacks.EarlyStopping('val_loss', patience=25, min_delta=0.001)

print('Begin training')
history = model.fit(train_data,
                    epochs=100, 
                    validation_data=val_data, 
                    verbose=1, 
                    steps_per_epoch=len(train_data),
                    callbacks=[early_stop]
                    )

# for id in range(593, 600):
#     im_in = val_data.scaler.transform(np.load(f'/home/jperez/data/sled250_2D/{id}.npy')).reshape(1, 89, 161, 1)
#     y_true = val_data.scaler.transform(np.load(f'/home/jperez/data/sled250_2D/{id+1}.npy'))

#     y_pred = model.predict(im_in).reshape(89, 161)
#     print(mean_squared_error(y_true, y_pred))