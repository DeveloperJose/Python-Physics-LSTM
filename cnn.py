from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow.keras as keras

np.set_printoptions(suppress=True, linewidth=np.inf)

class Sled2DGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, start, end, batch_size, shuffle):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.list_IDs = np.arange(start, end)
        self.on_epoch_end()

        self.scaler = StandardScaler()
        for id in tqdm(self.list_IDs, desc='Creating scaler'):
            im = np.load(self.data_dir / f'{id}.npy')
            self.scaler.partial_fit(im)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = np.zeros((self.batch_size, 89, 161, 1), dtype=np.float64)
        Y = np.zeros((self.batch_size, 89, 161, 1), dtype=np.float64)

        for idx, id in enumerate(list_IDs_temp):
            X[idx] = self.scaler.transform(np.load(self.data_dir / f'{id}.npy')).reshape(89, 161, 1)
            Y[idx] = self.scaler.transform(np.load(self.data_dir / f'{id+1}.npy')).reshape(89, 161, 1)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# sled250 = 1 to 741
print('Loading data')
train_data = Sled2DGenerator('/home/jperez/data/sled250_2D', 1, 593, batch_size=16, shuffle=True)
val_data = Sled2DGenerator('/home/jperez/data/sled250_2D', 593, 741, batch_size=16, shuffle=True)

print('Preparing model')
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(1, (3, 3), padding='same', input_shape=(89, 161, 1)))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
model.summary()

print('Begin training')
history = model.fit(train_data,
                    epochs=100, 
                    validation_data=val_data, 
                    verbose=1, 
                    steps_per_epoch=len(train_data))


for id in range(593, 600):
    im_in = val_data.scaler.transform(np.load(f'/home/jperez/data/sled250_2D/{id}.npy')).reshape(1, 89, 161, 1)
    y_true = val_data.scaler.transform(np.load(f'/home/jperez/data/sled250_2D/{id+1}.npy'))

    y_pred = model.predict(im_in).reshape(89, 161)
    print(mean_squared_error(y_true, y_pred))