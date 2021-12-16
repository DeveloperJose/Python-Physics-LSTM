import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import joblib
import datetime
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

N_CSV_ROWS = 14184
LOOKBACK = 10
BATCH_SIZE = 5024*4

CSV_HEADER = ['ID', 'X', 'Y', 'Z', 'ABS_P', 
                'AIR_D', 'AIR_F', 'AIR_F_X', 'AIR_F_Y', 
                'P', 'P_C', 'BULK_T', 'THERM_C', 'TOTAL_E', 'TOTAL_P', 'TOTAL_T', 
                'TURB_E_D', 'TURB_E_F', 'TURB_I', 'TURB_K_E', 
                'VEL_U', 'VEL_U_GRAD', 'VEL_U_GRAD_X', 'VEL_U_GRAD_Y',
                'VEL_V', 'VEL_V_GRAD', 'VEL_V_GRAD_X', 'VEL_V_GRAD_Y',
                'VORT', 'WATER_F', 'WATER_F_X', 'WATER_F_Y', 'WATER_H_F', 'WATER_I_E', 'WATER_M_F', 'WATER_V_F'
                ]
INPUTS = ['VEL_U']
OUTPUTS = ['VEL_U']

EXPERIMENT_N = 4

def read_csv(filename, scaler):
    df = pd.read_csv(filename, skiprows=5, header=0, names=CSV_HEADER)
    df = df.dropna()

    dx = []
    dy = []
    for x_name in INPUTS:
        dx.append(df[x_name].to_numpy())
    
    for y_name in OUTPUTS:
        dy.append(df[y_name].to_numpy())

    dx = np.array(dx).T
    dy = np.array(dy).T

    if scaler:
        dx = scaler.transform(dx)
    return dx, dy

def generate_permutations(n_timesteps, n_rows):
    return [(t, r) for t in np.arange(n_timesteps) for r in np.arange(n_rows)]

a = np.array(generate_permutations(5, 20))
b = np.array([x for x in itertools.product(range(5), range(20))])
print(np.array_equal(a, b))

# Scaler
# Recreating takes around 3 minutes
SCALER_PATH = 'scaler.pkl'
# SCALER_CREATION_DIRS = ['/home/jperez/datasets/sled250', '/home/jperez/datasets/sled300', '/home/jperez/datasets/sled350']
SCALER_CREATION_DIRS = ['/home/jperez/datasets/sled250']
if os.path.exists(SCALER_PATH):
    print('Loading previous scaler')
    SCALER = joblib.load(SCALER_PATH)
else:
    print('Recreating scaler')
    SCALER = StandardScaler()
    for creation_dir in SCALER_CREATION_DIRS:
        for filepath in Path(creation_dir).glob('*.csv'):
            X, Y = read_csv(filepath, None)
            SCALER.partial_fit(X)
    joblib.dump(SCALER, SCALER_PATH)

# Inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class SledDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, lookback, shuffle, start, end, step=10):
        print(f'Loading dataset {data_dir}')

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle
        self.start = start
        self.end = end
        self.step = step
        
        # Check if we have a serialized version of the data, if not, generate it
        n_count = ((end-start)//step)+1
        data_path = f'{self.data_dir.stem}_{start}_{end}_in_{INPUTS}_out_{OUTPUTS}.npz'
        if os.path.exists(data_path):
            csv_data = np.load(data_path)
            self.x_data = csv_data['X']
            self.y_data = csv_data['Y']
        else:
            self.x_data = []
            self.y_data = []

            for timestep in tqdm(range(start, end, step)):
                X, Y = self.__read_csv__(timestep)
                self.x_data.append(X)
                self.y_data.append(Y)
            self.x_data = np.array(self.x_data)
            self.y_data = np.array(self.y_data)
            np.savez(data_path, X=self.x_data, Y=self.y_data)

        # Some sanity checks
        assert self.x_data.shape[1] == self.y_data.shape[1], 'x_data and y_data have a shape mismatch in the number of CSV rows'
        assert self.x_data.shape[1] == N_CSV_ROWS, f'N_CSV_ROWS does not match {self.x_data.shape[1]}'

        # Helpful variables
        self.n_timesteps = self.x_data.shape[0]
        self.n_nodes = self.x_data.shape[1]

        # Generate a list of the valid timesteps for batches
        self.list_timesteps = np.arange(lookback, self.n_timesteps)
        self.list_rows = np.arange(N_CSV_ROWS)
        self.list_IDs = [(t, r) for t in self.list_timesteps for r in self.list_rows]
        # self.list_IDs = [x for x in itertools.product(self.list_timesteps, self.list_rows)]
        
        self.on_epoch_end()

    def __len__(self):
        # return int(np.floor(len(self.list_timesteps) / self.batch_size))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_pairs = [self.list_IDs[k] for k in batch_idxs]

        batch_x = []
        batch_y = []
        for (timestep, row) in batch_pairs:
            X = self.x_data[timestep-self.lookback:timestep, row]
            Y = self.y_data[timestep, row]

            batch_x.append(X)
            batch_y.append(Y)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __read_csv__(self, timestep):
        filename = os.path.join(self.data_dir, f'export{timestep}.csv')
        return read_csv(filename, SCALER)

# train_generator = SledDataGenerator('/home/jperez/datasets/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, start=10, end=7700+10)
# val_generator = SledDataGenerator('/home/jperez/datasets/sled300', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, start=10, end=7410+10)

train_generator = SledDataGenerator('/home/jperez/datasets/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, start=10, end=6100+10)
val_generator = SledDataGenerator('/home/jperez/datasets/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, start=6100, end=7700+10)

print('Preparing model')
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
model = keras.Sequential()

model.add(keras.layers.LSTM(128, input_shape=(LOOKBACK, len(INPUTS)), return_sequences=True, dropout=0.3))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(OUTPUTS))))

# model.add(keras.layers.LSTM(512, input_shape=(LOOKBACK, len(INPUTS)), return_sequences=True, dropout=0.1))
# model.add(keras.layers.LSTM(128, return_sequences=True, dropout=0.1))
# model.add(keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK)))
# model.add(keras.layers.Dense(len(OUTPUTS)))

model.compile(
    loss=keras.losses.MeanSquaredLogarithmicError(),
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.MeanSquaredError()]
)

model.summary()

print('Preparing checkpoints')
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
checkpoint = keras.callbacks.ModelCheckpoint(f'LSTM_v3_exp{EXPERIMENT_N}.hdf5',
                                             verbose=1,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/lstm_exp{EXPERIMENT_N}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

print('')
history = model.fit(
    x=train_generator,
    epochs=50,
    validation_data=val_generator,
    verbose=1,
    callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard],
    steps_per_epoch=len(train_generator)
)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

fig, ax = plt.subplots()
ax.plot(loss, label = 'train')
ax.plot(val_loss, label = 'val')
ax.set_title('Loss (Mean Squared Logarithmic Error)')
ax.legend(loc='upper right')

plt.savefig(f'lstm_exp_{EXPERIMENT_N}.png')