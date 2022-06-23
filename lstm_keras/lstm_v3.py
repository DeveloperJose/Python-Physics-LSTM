# Python native libraries
import os
import itertools
import joblib
from timeit import default_timer as timer
from pathlib import Path

# Science libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import neighbors
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ML libraries
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Other libraries
import matplotlib.pyplot as plt
from tqdm import tqdm
from twilio.rest import Client

# Project libraries
import keys
import data
import scaling

#%% Constants
EXPERIMENT_N = 29

LOOKBACK = 10
PREDICT_AHEAD = 1
NEIGHBOR_SIZE = 1

BATCH_SIZE = 5024

USE_2D = False

# INPUTS = ['X', 'Y', 'P', 'Vu', 'Vv', 'W.VF']
INPUTS = ['X', 'Y']
OUTPUTS = ['Vu']

SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}_2D={USE_2D}.pkl')
SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled255']

# No more scientific notation
np.set_printoptions(suppress=True, linewidth=np.inf)

#%% Build model from hyperparameters
def build_model(hp):
    pass

#%% Build custom tuner
# More documentation is available here: https://keras.io/guides/keras_tuner/custom_tuner/
class MyTuner(kt.Tuner):
    # You can set here any parameters from the fit() function if you want to use them
    # In our case, we want access to the training set generator (x) so that's the only one I'm including
    def run_trial(self, trial, x, *fit_args, **fit_kwargs):
        # Get the tuner hyperparameters from keras-tuner's API
        hp: kt.HyperParameters = trial.hyperparameters

        # Set some trial hyperparameters
        x.batch_size = hp.Int('batch_size', 32, 128, step=32, default=64) 

        # Let keras-tuner do the rest of the work
        super(MyTuner, self).run_trial(trial, x=x, *fit_args, **fit_kwargs)

if __name__ == '__main__':
    #%% Check if we forgot to update the experiment number
    assert not os.path.exists(f'lstm_exp_{EXPERIMENT_N}.png'), 'Experiment number already exists'

    #%% Twilio set-up
    client = Client(keys.account_sid, keys.auth_token)

    #%% Scaler set-up
    sc = scaling.load_or_create(SCALER_PATH, SCALER_CREATION_DIRS, INPUTS, OUTPUTS, USE_2D)

    #%% Data Loaders
    # train_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, 
    #                       shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=1, end=638+1)
    # val_generator = data.SledDataGenerator('/home/jperez/data/sled255', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=19, end=760+1)

    train_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, predict_ahead=PREDICT_AHEAD, neighbor_size=NEIGHBOR_SIZE, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, 
                                            start=1, end=510+1)

    train_generator.sciann = True

    val_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, predict_ahead=PREDICT_AHEAD, neighbor_size=NEIGHBOR_SIZE, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, 
                                            start=510, end=638+1)

    class Physics(keras.layers.Layer):
        def __init__(self):
            super(Physics, self).__init__()

        def call(self, inputs):
            print(inputs)
            # self.add_loss(keras.losses.mean_squared_error)
            return inputs
    #%% Model
    print('Preparing model')
    if USE_2D:
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(LOOKBACK, data.HEIGHT, data.WIDTH, len(INPUTS))))
        model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), return_sequences=True))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), return_sequences=True))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv3D(filters=2, kernel_size=(3,3,3), activation="sigmoid"))
        
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam()
        )
    else:
        # https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
        model = keras.Sequential()

        model.add(keras.layers.InputLayer(input_shape=(LOOKBACK, len(INPUTS))))
        model.add(Physics())
        model.add(keras.layers.LSTM(256, return_sequences=True))
        model.add(keras.layers.LSTM(128, return_sequences=True))
        # model.add(keras.layers.Dense(128, activation='relu'))
        # model.add(keras.layers.Dropout(0.1))
        # model.add(keras.layers.RepeatVector(PREDICT_AHEAD))
        # model.add(keras.layers.Reshape((PREDICT_AHEAD, len(INPUTS), 1, 64)))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK)))
        model.add(keras.layers.Flatten())
        # model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
        model.add(keras.layers.Dense(len(OUTPUTS)))

        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam()
        )

    model.summary()

    #%% Model checkpoints
    print('Preparing checkpoints')
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    checkpoint = keras.callbacks.ModelCheckpoint(f'LSTM_v3_exp{EXPERIMENT_N}.hdf5',
                                                verbose=1,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='auto')
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/lstm_exp{EXPERIMENT_N}')

    #%% Model training
    start_time = timer()
    print('Fitting model')
    history = model.fit(
        x=train_generator,
        epochs=50,
        validation_data=val_generator,
        verbose=1,
        callbacks=[early_stopping, checkpoint, tensorboard],
        steps_per_epoch=len(train_generator)
    )
    end_time = timer()
    duration = end_time-start_time
    print(f'Fitting took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h')

    #%% Model evaluation
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots()
    ax.plot(loss, label = 'train')
    ax.plot(val_loss, label = 'val')
    ax.set_title('Loss (Mean Squared Logarithmic Error)')
    ax.legend(loc='upper right')

    plt.savefig(f'lstm_exp_{EXPERIMENT_N}.png')

    # Send a text message via Twilio
    client.messages.create(
        body=f'Model {EXPERIMENT_N} has completed',
        from_=keys.src_phone,
        to=keys.dst_phone
    )