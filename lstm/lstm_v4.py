# Python native libraries
import os
import itertools
from pickletools import uint2
from unicodedata import ucd_3_2_0
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
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, losses

# tf.autograph.set_verbosity(3, True)

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# Other libraries
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from twilio.rest import Client

# Project libraries
import keys
import data
import scaling

# %% Constants
EXPERIMENT_N = 33

LOOKBACK = 10
PREDICT_AHEAD = 1
NEIGHBOR_SIZE = 1

BATCH_SIZE = 5024*8

USE_2D = False

INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
# OUTPUTS = ['Vu', 'Vv']
OUTPUTS = ['Vu', 'Vv']

SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}_2D={USE_2D}.pkl')
SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled255']

# No more scientific notation
np.set_printoptions(suppress=True, linewidth=np.inf)

# %% Build model from hyperparameters
def build_model(hp):
    pass

# %% Build custom tuner
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
    # %% Check if we forgot to update the experiment number
    assert not os.path.exists(f'lstm_exp_{EXPERIMENT_N}.png'), 'Experiment number already exists'

    # %% Twilio set-up
    client = Client(keys.account_sid, keys.auth_token)

    # %% Scaler set-up
    sc = scaling.load_or_create(SCALER_PATH, SCALER_CREATION_DIRS, INPUTS, OUTPUTS, USE_2D)

    # %% Data Loaders
    # train_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK,
    #                       shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=1, end=638+1)
    # val_generator = data.SledDataGenerator('/home/jperez/data/sled255', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=19, end=760+1)

    train_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, predict_ahead=PREDICT_AHEAD, neighbor_size=NEIGHBOR_SIZE, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc,
                                             start=1, end=510+1)
    train_generator.sciann = True

    val_generator = data.SledDataGenerator('/home/jperez/data/sled250', batch_size=BATCH_SIZE, lookback=LOOKBACK, predict_ahead=PREDICT_AHEAD, neighbor_size=NEIGHBOR_SIZE, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc,
                                           start=510, end=638+1)

    val_generator.sciann = True

    def lambda_d1(tensor):
        xyt = tensor[0]
        psi_p = tensor[1]

        psi = psi_p[:, 0]
        p = psi_p[:, 1]

        psi_grad = K.gradients(psi, xyt)
        p_grad = K.gradients(p, xyt)
        if len(psi_grad) > 0 and psi_grad[0] != None:
            # [Batch, Lookback, Input]
            v = -psi_grad[0][:, -1, 0]
            u = psi_grad[0][:, -1, 1]

            p_x = p_grad[0][:, -1, 0]
            p_y = p_grad[0][:, -1, 0]
            return tf.stack([u, v, p_x, p_y], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return tf.stack([z, z, z, z], axis=1)

    def lambda_d2(tensor):
        xyt = tensor[0]
        d1 = tensor[1]

        u_grad = K.gradients(d1[0], xyt)[0]
        v_grad = K.gradients(d1[1], xyt)[0]
        if u_grad is not None:
            u_x = u_grad[:, -1, 0]
            u_y = u_grad[:, -1, 1]
            u_t = u_grad[:, -1, 2]

            v_x = v_grad[:, -1, 0]
            v_y = v_grad[:, -1, 1]
            v_t = v_grad[:, -1, 2]

            return tf.stack([u_x, u_y, u_t, v_x, v_y, v_t], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return tf.stack([z, z, z, z, z, z], axis=1)

    def lambda_d3(tensor):
        xyt = tensor[0]
        d2 = tensor[1]

        u_x_grad = K.gradients(d2[0], xyt)[0]
        u_y_grad = K.gradients(d2[1], xyt)[0]
        v_x_grad = K.gradients(d2[3], xyt)[0]
        v_y_grad = K.gradients(d2[4], xyt)[0]
        if u_x_grad is not None:
            u_xx = u_x_grad[:, -1, 0]
            u_yy = u_y_grad[:, -1, 1]

            v_xx = v_x_grad[:, -1, 0]
            v_yy = v_y_grad[:, -1, 1]
            return tf.stack([u_xx, u_yy, v_xx, v_yy], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return tf.stack([z, z, z, z], axis=1)

    def lambda_output(tensor):
        d1 = tensor[0]
        d2 = tensor[1]
        d3 = tensor[2]
        lstm_output = tensor[3]

        u, v, p_x, p_y = d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3]
        u_x, u_y, u_t, v_x, v_y, v_t = d2[:, 0], d2[:, 1], d2[:, 2], d2[:, 3], d2[:, 4], d2[:, 5]
        u_xx, u_yy, v_xx, v_yy = d3[:, 0], d3[:, 1], d3[:, 2], d3[:, 3]

        lstm_u = lstm_output[:, 0]
        lstm_v = lstm_output[:, 1]
        l1 = 0.999
        l2 = 0.01

        f_u = u_t + l1*(u*u_x + v*u_y) + p_x - l2*(u_xx + u_yy)
        f_v = v_t + l1*(u*v_x + v*v_y) + p_y - l2*(v_xx + v_yy)

        lstm_w = 0.8
        pinns_w = 1 - lstm_w

        return tf.stack([lstm_w*lstm_u + pinns_w*u, lstm_w*lstm_v+pinns_w*v, f_u, f_v], axis=1)
    
    # %% Model
    print('Preparing Model')
    input_layer = keras.layers.Input(shape=(LOOKBACK, len(INPUTS)))

    # Dense Branch
    # For input, take the current timestep and X,Y,T (recall layer[Batch_Size, Timestep, Input])
    # prev_dense = input_layer[:, -1, :3]
    dense1 = keras.layers.Dense(20)(input_layer[:, -1, :3])
    dense2 = keras.layers.Dense(20)(dense1)
    dense_output = keras.layers.Dense(2, name='Dense_Output')(dense2)

    # LSTM Branch
    lstm1 = keras.layers.LSTM(256, return_sequences=True)(input_layer)
    lstm2 = keras.layers.LSTM(128, return_sequences=True)(lstm1)
    lstm_td = keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK))(lstm2)
    lstm_flat = keras.layers.Flatten()(lstm_td)
    lstm_output = keras.layers.Dense(len(OUTPUTS), name='LSTM_Output')(lstm_flat)

    # Dense Lambdas
    pinns1 = keras.layers.Lambda(lambda_d1)([input_layer, dense_output])
    pinns2 = keras.layers.Lambda(lambda_d2)([input_layer, pinns1])
    pinns3 = keras.layers.Lambda(lambda_d3)([input_layer, pinns2])

    # PINNs/Combination Lambda
    pinns_output = keras.layers.Lambda(lambda_output)([pinns1, pinns2, pinns3, lstm_output])
    lstm_model = keras.Model(input_layer, pinns_output)
    lstm_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(reduction=losses.Reduction.AUTO))
    lstm_model.summary()

    keras.utils.plot_model(lstm_model, to_file='model.png', show_shapes=True)

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
    history = lstm_model.fit(
        x=train_generator,
        epochs=1,
        validation_data=val_generator,
        verbose=1,
        callbacks=[early_stopping, checkpoint, tensorboard],
        steps_per_epoch=len(train_generator)
    )
    end_time = timer()
    duration = end_time-start_time
    print(f'Fitting took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h')

    # #%% Model evaluation
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # fig, ax = plt.subplots()
    # ax.plot(loss, label = 'train')
    # ax.plot(val_loss, label = 'val')
    # ax.set_title('Loss (Mean Squared Logarithmic Error)')
    # ax.legend(loc='upper right')

    # #%% Individual MSEs
    # from sklearn.metrics import mean_squared_error
    # y_true = []
    # y_pred = []
    # with tqdm(total=len(val_generator), desc='Computing MSEs') as p_bar:
    #     for batch_x, batch_y in val_generator:
    #         pred = lstm_model.predict(batch_x)
    #         y_pred.extend(pred)
    #         y_true.extend(batch_y)
    #         p_bar.update()
    #         # p_bar.postfix = f'MSE: {mean_squared_error(y_true, y_pred, multioutput="raw_values")}'

    # print('Final', mean_squared_error(y_true, y_pred, multioutput='raw_values'))
            
    # plt.savefig(f'lstm_exp_{EXPERIMENT_N}.png')

    # #%% Send a text message via Twilio
    # client.messages.create(
    #     body=f'Model {EXPERIMENT_N} has completed with MSES {mean_squared_error(y_true, y_pred, multioutput="raw_values")}',
    #     from_=keys.src_phone,
    #     to=keys.dst_phone
    # )