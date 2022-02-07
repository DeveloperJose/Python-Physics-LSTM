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
EXPERIMENT_N = 29

LOOKBACK = 10
PREDICT_AHEAD = 1
NEIGHBOR_SIZE = 1

BATCH_SIZE = 5024

USE_2D = False

INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
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

    # def custom_loss(input_tensor: tf.Tensor, output_tensor: tf.Tensor):
    #     def loss(y_true, y_pred):
    #         psi = output_tensor[0][:, 0:1]
    #         p = output_tensor[0][:, 1:2]
    #         # import pdb
    #         # pdb.set_trace()
    #         # x = input_tensor[:, 0]
    #         # y = input_tensor[:, 1]
    #         # t = input_tensor[:, 2]

    #         # psi = output_tensor[:, 0]
    #         # p = output_tensor[:, 1]

    #         # u = K.gradients(psi, y)[0]
    #         # v = -K.gradients(psi, x)[0]
    #         # print('grads', grads, grads.shape)
    #         return K.square()
    #     return loss

    # class GradientLayer(keras.layers.Layer):
    #     def __init__(self, model, **kwargs):
    #         super().__init__(**kwargs)
    #         self.model = model

    #     def call(self, xyt):
    #         print('******************* CALL')
    #         x, y, t = [ xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1]) ]
    #         with tf.GradientTape(persistent=True) as t3:
    #             t3.watch([x, y, t])
    #             with tf.GradientTape(persistent=True) as t2:
    #                 t2.watch([x, y, t])
    #                 with tf.GradientTape(persistent=True) as t1:
    #                     t1.watch([x, y, t])
    #                     # Pass input to LSTM model
    #                     psi_p = self.model(tf.concat([x, y, t], axis=-1))
    #                     # Get outputs from LSTM model
    #                     psi = psi_p[..., 0, tf.newaxis]
    #                     p   = psi_p[..., 1, tf.newaxis]

    #                 # Gradient of [PSI,P] w.r.t inputs [X,Y]
    #                 # Only use the "current timestep" for X/Y
    #                 # v = t1.batch_jacobian(psi, x[:, -1])
    #                 # u = t1.batch_jacobian(psi, y[:, -1])

    #                 # p_x = t1.batch_jacobian(p, x[:, -1])

    #                 # p_y = t1.batch_jacobian(p, y[:, -1])
    #                 v = -tf.reshape(t1.batch_jacobian(psi, x[:, -1]), [-1, 1])
    #                 u = tf.reshape(t1.batch_jacobian(psi, y[:, -1]), [-1, 1])

    #                 p_x = tf.reshape(t1.batch_jacobian(p, x[:, -1]), [-1, 1])
    #                 p_y = tf.reshape(t1.batch_jacobian(p, y[:, -1]), [-1, 1])

    #             # Gradients of [U,V] w.r.t inputs [X,Y,T]
    #             # u_x = t2.batch_jacobian(u, x[:, -1])
    #             # u_y = t2.batch_jacobian(u, y[:, -1])
    #             # u_t = t2.batch_jacobian(u, t[:, -1])

    #             # v_x = t2.batch_jacobian(v, x[:, -1])
    #             # v_y = t2.batch_jacobian(v, y[:, -1])
    #             # v_t = t2.batch_jacobian(v, t[:, -1])
    #             u_x = tf.reshape(t2.batch_jacobian(u, x[:, -1]), [-1, 1])
    #             u_y = tf.reshape(t2.batch_jacobian(u, y[:, -1]), [-1, 1])
    #             u_t = tf.reshape(t2.batch_jacobian(u, t[:, -1]), [-1, 1])

    #             v_x = tf.reshape(t2.batch_jacobian(v, x[:, -1]), [-1, 1])
    #             v_y = tf.reshape(t2.batch_jacobian(v, y[:, -1]), [-1, 1])
    #             v_t = tf.reshape(t2.batch_jacobian(v, t[:, -1]), [-1, 1])

    #         # Gradients of [U_X, U_Y, V_X, V_Y] w.r.t inputs [X,Y]
    #         # u_xx = t3.batch_jacobian(u_x, x[:, -1])
    #         # u_yy = t3.batch_jacobian(u_y, y[:, -1])

    #         # v_xx = t3.batch_jacobian(v_x, x[:, -1])
    #         # v_yy = t3.batch_jacobian(v_y, y[:, -1])
    #         u_xx = tf.reshape(t3.batch_jacobian(u_x, x[:, -1]), [-1, 1])
    #         u_yy = tf.reshape(t3.batch_jacobian(u_y, y[:, -1]), [-1, 1])

    #         v_xx = tf.reshape(t3.batch_jacobian(v_x, x[:, -1]), [-1, 1])
    #         v_yy = tf.reshape(t3.batch_jacobian(v_y, y[:, -1]), [-1, 1])

    #         # Clean-up tapes
    #         del t1, t2, t3

    #         p_grads = p, p_x, p_y
    #         u_grads = u, u_x, u_y, u_t, u_xx, u_yy
    #         v_grads = v, v_x, v_y, v_t, v_xx, v_yy

    #         lambda1 = 0.999
    #         lambda2 = 0.01
    #         f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
    #         f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

    #         # return tf.concat([u, v, f_u, f_v])

    def lambda_test(tensor):
        xyt = tensor[0]
        psi_p = tensor[1]

        x, y, t = [xyt[..., i, tf.newaxis] for i in range(xyt.shape[-1])]
        psi = psi_p[..., 0, tf.newaxis]
        p = psi_p[..., 1, tf.newaxis]
        pdb.set_trace()
        with tf.GradientTape(persistent=True) as t3:
            t3.watch([x, y, t, psi, p])
            with tf.GradientTape(persistent=True) as t2:
                t2.watch([x, y, t, psi, p])
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch([x, y, t, psi, p])
                    # Pass input to LSTM model
                    # psi_p = self.model(tf.concat([x, y, t], axis=-1))
                    # Get outputs from LSTM model
                    pdb.set_trace()

                # Gradient of [PSI,P] w.r.t inputs [X,Y]
                # Only use the "current timestep" for X/Y
                # v = t1.batch_jacobian(psi, x[:, -1])
                # u = t1.batch_jacobian(psi, y[:, -1])

                # p_x = t1.batch_jacobian(p, x[:, -1])

                # p_y = t1.batch_jacobian(p, y[:, -1])
                v = -tf.reshape(t1.batch_jacobian(psi, x[:, -1]), [-1, 1])
                # u = tf.reshape(t1.batch_jacobian(psi, y[:, -1]), [-1, 1])
                u = t1.gradient(psi, y)
                pdb.set_trace()

                # p_x = tf.reshape(t1.batch_jacobian(p, x[:, -1]), [-1, 1])
                # p_y = tf.reshape(t1.batch_jacobian(p, y[:, -1]), [-1, 1])

            # Gradients of [U,V] w.r.t inputs [X,Y,T]
            # u_x = t2.batch_jacobian(u, x[:, -1])
            # u_y = t2.batch_jacobian(u, y[:, -1])
            # u_t = t2.batch_jacobian(u, t[:, -1])

            # v_x = t2.batch_jacobian(v, x[:, -1])
            # v_y = t2.batch_jacobian(v, y[:, -1])
            # v_t = t2.batch_jacobian(v, t[:, -1])

            # u_x = tf.reshape(t2.batch_jacobian(u, x[:, -1]), [-1, 1])
            # u_y = tf.reshape(t2.batch_jacobian(u, y[:, -1]), [-1, 1])
            # u_t = tf.reshape(t2.batch_jacobian(u, t[:, -1]), [-1, 1])

            # v_x = tf.reshape(t2.batch_jacobian(v, x[:, -1]), [-1, 1])
            # v_y = tf.reshape(t2.batch_jacobian(v, y[:, -1]), [-1, 1])
            # v_t = tf.reshape(t2.batch_jacobian(v, t[:, -1]), [-1, 1])

        # Gradients of [U_X, U_Y, V_X, V_Y] w.r.t inputs [X,Y]
        # u_xx = t3.batch_jacobian(u_x, x[:, -1])
        # u_yy = t3.batch_jacobian(u_y, y[:, -1])

        # v_xx = t3.batch_jacobian(v_x, x[:, -1])
        # v_yy = t3.batch_jacobian(v_y, y[:, -1])

        # u_xx = tf.reshape(t3.batch_jacobian(u_x, x[:, -1]), [-1, 1])
        # u_yy = tf.reshape(t3.batch_jacobian(u_y, y[:, -1]), [-1, 1])

        # v_xx = tf.reshape(t3.batch_jacobian(v_x, x[:, -1]), [-1, 1])
        # v_yy = tf.reshape(t3.batch_jacobian(v_y, y[:, -1]), [-1, 1])

        # Clean-up tapes
        del t1, t2, t3

        # p_grads = p, p_x, p_y
        # u_grads = u, u_x, u_y, u_t, u_xx, u_yy
        # v_grads = v, v_x, v_y, v_t, v_xx, v_yy

        # lambda1 = 0.999
        # lambda2 = 0.01
        # f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
        # f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

        # return tf.concat([u, v, f_u, f_v], axis=1)
        print('Call = ', u)
        return u

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

        return tf.stack([0.5*lstm_u + 0.5*u, 0.5*lstm_v+0.5*v, f_u, f_v], axis=1)
    # %% Model
    print('Initializing')
    # https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

    # input_tensor = layers.Input(shape=(LOOKBACK, len(INPUTS)))
    # lstm1 = layers.LSTM(256, return_sequences=True)(input_tensor)
    # lstm2 = layers.LSTM(128, return_sequences=True)(lstm1)
    # td = layers.TimeDistributed(keras.layers.Dense(LOOKBACK))(lstm2)
    # f = layers.Flatten()(td)
    # output_tensor = layers.Dense(2)(f)

    # model = keras.Model(input_tensor, output_tensor)

    print('Preparing LSTM')
    input_layer = keras.layers.Input(shape=(LOOKBACK, len(INPUTS)))

    lstm1 = keras.layers.LSTM(256, return_sequences=True)(input_layer)
    lstm2 = keras.layers.LSTM(128, return_sequences=True)(lstm1)
    lstm_td = keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK))(lstm2)
    lstm_flat = keras.layers.Flatten()(lstm_td)
    lstm_output = keras.layers.Dense(len(OUTPUTS), name='LSTM_Output')(lstm_flat)

    dense1 = keras.layers.Dense(20)(input_layer)
    dense2 = keras.layers.Dense(20)(dense1)
    dense_output = keras.layers.Dense(2, name='Dense_Output')(dense2)

    pinns1 = keras.layers.Lambda(lambda_d1)([input_layer, dense_output])
    pinns2 = keras.layers.Lambda(lambda_d2)([input_layer, pinns1])
    pinns3 = keras.layers.Lambda(lambda_d3)([input_layer, pinns2])
    pinn_output = keras.layers.Lambda(lambda_output)([pinns1, pinns2, pinns3, lstm_output])
    lstm_model = keras.Model(input_layer, pinn_output)
    lstm_model.summary()

    keras.utils.plot_model(lstm_model, to_file='model.png', show_shapes=True)

    # Physics-Informed Part
    # print('Preparing GradientLayer')
    # grads = GradientLayer(lstm_model)

    # print('Preparing Physics-Informed Model')
    # # input_layer2 = keras.layers.Input(shape=(1, len(INPUTS)))

    # print('Calling grads')

    # psi_grads, p_grads, u_grads, v_grads = grads(input_layer)
    # p, p_x, p_y = p_grads
    # u, u_x, u_y, u_t, u_xx, u_yy = u_grads
    # v, v_x, v_y, v_t, v_xx, v_yy = v_grads

    # print('Setting up outputs')
    # f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
    # f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)
    # pinn_model = keras.Model(input_layer, outputs=tf.concat([u, v, f_u, f_v], axis=1))

    # %% Training
    print('Compiling model')
    opt = keras.optimizers.Adam()
    loss_fn = keras.losses.MeanSquaredError()
    # train_metric = keras.metrics.MeanSquaredError()
    # val_metric = keras.metrics.MeanSquaredError()
    lstm_model.compile(optimizer=opt, loss=loss_fn)

    print('Training model')
    # model.compile(loss=custom_loss(model.inputs, model.outputs), optimizer=opt, experimental_run_tf_function=False)
    # lstm_model.summary()

    history = lstm_model.fit(
        x=train_generator,
        epochs=1,
        validation_data=val_generator,
        verbose=1,
        # callbacks=[early_stopping, checkpoint, tensorboard],
        steps_per_epoch=len(train_generator)
    )
    # for epoch in range(1):
    #     # Train
    #     start_time = timer()
    #     with tqdm(total=len(train_generator), desc=f'Epoch {epoch}') as pbar:
    #         for batch_x, batch_y in train_generator:
    #             with tf.GradientTape(persistent=True) as tape:
    #                 pred = model(batch_x, training=True)
    #                 psi = pred[:, 0]
    #                 p = pred[:, 1]
    #                 # loss_val = loss_fn(batch_y, pred)

    #             # grads = tape.gradient(loss_val, model.trainable_variables)
    #             import pdb
    #             pdb.set_trace()
    #             opt.apply_gradients(zip(grads, model.trainable_variables))
    #             train_metric.update_state(batch_y, pred)
    #             pbar.update()
    #             pbar.postfix = f'Train MSE: {train_metric.result()}'

    #     epoch_duration = timer() - start_time

    #     # Val
    #     start_time = timer()
    #     for batch_x, batch_y in tqdm(val_generator, desc='Validation'):
    #         pred = model(batch_x, training=False)
    #         val_metric.update_state(batch_y, pred)

    #     val_duration = timer() - start_time

    #     # Stats
    #     print(f'Epoch {epoch} took {epoch_duration:.3f}sec = {epoch_duration/60:.3f}min = {epoch_duration/60/60:.3f}h | Validation took {val_duration:.3f}sec | Train MSE={train_metric.result()} | Validation MSE={val_metric.result()}')

    #     train_generator.on_epoch_end()
    #     val_generator.on_epoch_end()
    #     train_metric.reset_states()
    #     val_metric.reset_states()

    # model.summary()

    # #%% Model checkpoints
    # print('Preparing checkpoints')
    # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    # checkpoint = keras.callbacks.ModelCheckpoint(f'LSTM_v3_exp{EXPERIMENT_N}.hdf5',
    #                                             verbose=1,
    #                                             monitor='val_loss',
    #                                             save_best_only=True,
    #                                             mode='auto')
    # tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/lstm_exp{EXPERIMENT_N}')

    # #%% Model training
    # start_time = timer()
    # print('Fitting model')
    # history = model.fit(
    #     x=train_generator,
    #     epochs=50,
    #     validation_data=val_generator,
    #     verbose=1,
    #     callbacks=[early_stopping, checkpoint, tensorboard],
    #     steps_per_epoch=len(train_generator)
    # )
    # end_time = timer()
    # duration = end_time-start_time
    # print(f'Fitting took {duration:.3f}sec = {duration/60:.3f}min = {duration/60/60:.3f}h')

    # #%% Model evaluation
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # fig, ax = plt.subplots()
    # ax.plot(loss, label = 'train')
    # ax.plot(val_loss, label = 'val')
    # ax.set_title('Loss (Mean Squared Logarithmic Error)')
    # ax.legend(loc='upper right')

    # plt.savefig(f'lstm_exp_{EXPERIMENT_N}.png')

    # Send a text message via Twilio
    client.messages.create(
        body=f'Model {EXPERIMENT_N} has completed',
        from_=keys.src_phone,
        to=keys.dst_phone
    )
