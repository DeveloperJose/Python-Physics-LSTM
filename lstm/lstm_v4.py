# Python native libraries
import pdb
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
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, losses

# tf.autograph.set_verbosity(3, True)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Other libraries
from twilio.rest import Client
from tqdm import tqdm

# Project libraries
import scaling
import data
import keys

# %% Constants
EXPERIMENT_N = 36

LOOKBACK = 10
PREDICT_AHEAD = 1
NEIGHBOR_SIZE = 1

BATCH_SIZE = 5024*8

USE_2D = False

INPUTS = ['X', 'Y', 'T', 'Vu', 'Vv', 'P', 'W.VF']
OUTPUTS = ['Vu']

SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}_2D={USE_2D}.pkl')
SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled255']

# No more scientific notation
np.set_printoptions(suppress=True, linewidth=np.inf)

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

    # def lambda_master(tensor):
    #     xyt = tensor[0]
    #     psi_p = tensor[1]
    #     lstm_output = tensor[2]

    #     # Parameters
    #     l1 = 0.999
    #     l2 = 0.01
    #     lstm_w = 0.8
    #     pinns_w = 1 - lstm_w

    #     # Split tensors into individual pieces
    #     psi = psi_p[:, 0]
    #     p = psi_p[:, 1]

    #     lstm_u = lstm_output[:, 0]
    #     lstm_v = lstm_output[:, 1]

    #     # d1 = u, v, p_x, p_y
    #     psi_grad = K.gradients(psi, xyt)
    #     p_grad = K.gradients(p, xyt)
    #     if len(psi_grad) > 0 and psi_grad[0] != None:
    #         # [Batch, Lookback, Input (X, Y, T, ...)]
    #         print('Testing D1?')
    #         v = -psi_grad[0][:, -1, 0]
    #         u = psi_grad[0][:, -1, 1]

    #         p_x = p_grad[0][:, -1, 0]
    #         p_y = p_grad[0][:, -1, 0]
    #         d1 = K.stack([u, v, p_x, p_y], axis=1)
    #         print('v1=', K.gradients(d1[0], xyt)[0], 'v2=', K.gradients(u, xyt), 'v3=', K.gradients(K.gradients(psi, xyt), xyt))
    #     else:
    #         print('D1 Zeros', psi_grad)
    #         z = K.zeros_like(xyt)[:, 0, 0]
    #         d1 = K.stack([z, z, z, z], axis=1)
    #         u, v, p_x, p_y = z, z, z, z

    #     # d2 = u_x, u_y, u_t, v_x, v_y, v_t
    #     u_grad = K.gradients(d1[0], xyt)[0]
    #     v_grad = K.gradients(d1[1], xyt)[0]
    #     if u_grad is not None:
    #         u_x = u_grad[:, -1, 0]
    #         u_y = u_grad[:, -1, 1]
    #         u_t = u_grad[:, -1, 2]

    #         v_x = v_grad[:, -1, 0]
    #         v_y = v_grad[:, -1, 1]
    #         v_t = v_grad[:, -1, 2]

    #         d2 = K.stack([u_x, u_y, u_t, v_x, v_y, v_t], axis=1)
    #     else:
    #         print('D2 Zeros')
    #         z = K.zeros_like(xyt)[:, 0, 0]
    #         d2 = K.stack([z, z, z, z, z, z], axis=1)
    #         u_x, u_y, u_t, v_x, v_y, v_t = z, z, z, z, z, z

    #     # d3 = u_xx, u_yy, v_xx, v_yy
    #     u_x_grad = K.gradients(d2[0], xyt)[0]
    #     u_y_grad = K.gradients(d2[1], xyt)[0]
    #     v_x_grad = K.gradients(d2[3], xyt)[0]
    #     v_y_grad = K.gradients(d2[4], xyt)[0]
    #     if u_x_grad is not None:
    #         print('Testing D3')
    #         u_xx = u_x_grad[:, -1, 0]
    #         u_yy = u_y_grad[:, -1, 1]

    #         v_xx = v_x_grad[:, -1, 0]
    #         v_yy = v_y_grad[:, -1, 1]
    #         d3 = K.stack([u_xx, u_yy, v_xx, v_yy], axis=1)
    #     else:
    #         print('D3 Zeros')
    #         z = K.zeros_like(xyt)[:, 0, 0]
    #         d3 = K.stack([z, z, z, z], axis=1)
    #         u_xx, u_yy, v_xx, v_yy = z, z, z, z

    #     # output
    #     f_u = u_t + l1*(u*u_x + v*u_y) + p_x - l2*(u_xx + u_yy)
    #     f_v = v_t + l1*(u*v_x + v*v_y) + p_y - l2*(v_xx + v_yy)
    #     return K.stack([lstm_w*lstm_u + pinns_w*u, lstm_w*lstm_v+pinns_w*v, f_u, f_v], axis=1)

    def lambda_d1(tensor):
        xyt = tensor[0]
        psi_p = tensor[1]

        psi = psi_p[:, 0]
        p = psi_p[:, 1]

        psi_grad = K.gradients(psi, xyt)
        p_grad = K.gradients(p, xyt)
        if len(psi_grad) > 0 and psi_grad[0] != None:
            print('Valid D1')
            # [Batch, Lookback, Input (X, Y, T, ...)]
            v = -psi_grad[0][:, -1, 0]
            u = psi_grad[0][:, -1, 1]

            p_x = p_grad[0][:, -1, 0]
            p_y = p_grad[0][:, -1, 0]
            return K.stack([u, v, p_x, p_y], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return K.stack([z, z, z, z], axis=1)

    def lambda_d2(tensor):
        xyt = tensor[0]
        d1 = tensor[1]

        u_grad = K.gradients(d1[0], xyt)[0]
        v_grad = K.gradients(d1[1], xyt)[0]
        if u_grad is not None:
            print('Valid D2')
            u_x = u_grad[:, -1, 0]
            u_y = u_grad[:, -1, 1]
            u_t = u_grad[:, -1, 2]

            v_x = v_grad[:, -1, 0]
            v_y = v_grad[:, -1, 1]
            v_t = v_grad[:, -1, 2]

            return K.stack([u_x, u_y, u_t, v_x, v_y, v_t], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return K.stack([z, z, z, z, z, z], axis=1)

    def lambda_d3(tensor):
        xyt = tensor[0]
        d2 = tensor[1]

        u_x_grad = K.gradients(d2[0], xyt)[0]
        u_y_grad = K.gradients(d2[1], xyt)[0]
        v_x_grad = K.gradients(d2[3], xyt)[0]
        v_y_grad = K.gradients(d2[4], xyt)[0]
        if u_x_grad is not None:
            print('Valid D3')
            u_xx = u_x_grad[:, -1, 0]
            u_yy = u_y_grad[:, -1, 1]

            v_xx = v_x_grad[:, -1, 0]
            v_yy = v_y_grad[:, -1, 1]
            return K.stack([u_xx, u_yy, v_xx, v_yy], axis=1)
        z = K.zeros_like(xyt)[:, 0, 0]
        return K.stack([z, z, z, z], axis=1)

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

        return K.stack([lstm_w*lstm_u + pinns_w*u, lstm_w*lstm_v+pinns_w*v, f_u, f_v], axis=1)

    # class PINNSLayer(keras.layers.Layer):
    #     def __init__(self):
    #         super(PINNSLayer, self).__init__()
    #         self.pinns1 = keras.layers.Lambda(lambda_d1)

    #     def call(self, inputs):
    #         input_layer = inputs[0]
    #         dense_output = inputs[1]
    #         lstm_output = inputs[2]
    # %% Model
    print('Preparing Model')
    input_shape = (LOOKBACK, len(INPUTS))
    input_layer = keras.layers.Input(shape=input_shape, name='input')

    # # Dense Branch
    # # For input, take the current timestep and X,Y,T (recall layer[Batch_Size, Timestep, Input])
    # # prev_dense = input_layer[:, -1, :3]
    # # dense1 = keras.layers.Dense(20)(input_layer[:, -1, :3])
    # dense1 = keras.layers.Dense(20)(input_layer)
    # dense2 = keras.layers.Dense(20)(dense1)
    # dense_output = keras.layers.Dense(2, name='dense_output')(dense2)

    # # LSTM Branch
    # lstm1 = keras.layers.LSTM(256, return_sequences=True)(input_layer)
    # lstm2 = keras.layers.LSTM(128, return_sequences=True)(lstm1)
    # lstm_td = keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK))(lstm2)
    # lstm_flat = keras.layers.Flatten()(lstm_td)
    # lstm_output = keras.layers.Dense(len(OUTPUTS), name='lstm_output')(lstm_flat)

    # # Dense Lambdas
    # pinns1 = keras.layers.Lambda(lambda_d1)([input_layer, dense_output])
    # pinns2 = keras.layers.Lambda(lambda_d2)([input_layer, pinns1])
    # pinns3 = keras.layers.Lambda(lambda_d3)([input_layer, pinns2])
    # pinns_output = keras.layers.Lambda(lambda_output)([pinns1, pinns2, pinns3, lstm_output])
    # # pinns_output = keras.layers.Lambda(lambda_master)([input_layer, dense_output, lstm_output])

    # pinns1 = LambD1()([input_layer, dense_output])
    # pinns2 = LambD2()([input_layer, pinns1])
    # pinns3 = LambD3()([input_layer, pinns2])
    # pinns_output = LambOutput()([pinns1, pinns2, pinns3, lstm_output])

    # lstm_model = keras.Model(input_layer, pinns_output)
    # lstm_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(reduction=losses.Reduction.AUTO))
    # lstm_model.summary()

    # ********************************************** DEBUGGING: LSTM
    # train_generator.sciann = False
    # val_generator.sciann = False
    # lstm1 = keras.layers.LSTM(256, return_sequences=True)(input_layer)
    # lstm2 = keras.layers.LSTM(128, return_sequences=True)(lstm1)
    # lstm_td = keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK))(lstm2)
    # lstm_flat = keras.layers.Flatten()(lstm_td)
    # lstm_output = keras.layers.Dense(len(OUTPUTS), name='LSTM_Output')(lstm_flat)
    # lstm_model = keras.Model(input_layer, lstm_output)
    # lstm_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(reduction=losses.Reduction.AUTO))
    # lstm_model.summary()

    # keras.utils.plot_model(lstm_model, to_file='model.png', show_shapes=True)

    # %% Model checkpoints
    # print('Preparing checkpoints')
    # early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    # checkpoint = keras.callbacks.ModelCheckpoint(f'LSTM_v3_exp{EXPERIMENT_N}.hdf5',
    #                                             verbose=1,
    #                                             monitor='val_loss',
    #                                             save_best_only=True,
    #                                             mode='auto')
    # tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/lstm_exp{EXPERIMENT_N}')

    # ********************************************* DEBUGGING
    dense_m = keras.models.Sequential(layers=[
        input_layer,
        keras.layers.Dense(20),
        keras.layers.Dense(20),
        keras.layers.Dense(2, name='dense_output')
    ])

    lstm_m = keras.models.Sequential(layers=[
        input_layer,
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(LOOKBACK)),
        keras.layers.Flatten(),
        keras.layers.Dense(len(OUTPUTS), name='lstm_output')
    ])

    class FirstDerivativeLayer(keras.layers.Layer):
        def __init__(self, dense_model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dense_model = dense_model

        def call(self, inputs, training=None, mask=None):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                # Pass inputs to dense_model to get psi and p
                psi_p = self.dense_model(inputs, training=training, mask=mask)
                psi = psi_p[:, 0]
                p = psi_p[:, 1]

            psi_grads = tape.gradient(psi, inputs)
            p_grads = tape.gradient(p, inputs)
            u = psi_grads[:, :, 1]
            v = psi_grads[:, :, 0]
            p_x = p_grads[:, :, 0]
            p_y = p_grads[:, :, 1]
            return u

    def F_Model(data):
        x = tf.Variable(data[:, :, 0])
        y = tf.Variable(data[:, :, 1])
        t = tf.Variable(data[:, :, 2])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, t])
            psi_p = dense_m(data)
            psi = psi_p[:, 0]
            p = psi_p[:, 1]

        pdb.set_trace()
        psi_grads = tape.gradient(psi, data)

    class PINNS(keras.Model):
        def __init__(self, dense_model, lstm_model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dense_model = dense_model
            # self.lstm_model = lstm_model

            self.first_lambda = FirstDerivativeLayer(dense_model)

        def find_grad(self, f1, f2):
            return keras.layers.Lambda(lambda x: K.gradients(x[0], x[1]))([f1, f2])

        def train_step(self, data):
            batch_x, batch_y = data
            # var1 = tf.Variable(batch_x, dtype=tf.float32, name='batch_data')
            # var2 = tf.Variable(batch_x[:, :, 0], dtype=tf.float32, name='batch_x')
            # with tf.GradientTape(persistent=True) as tape2:
            #     tape2.watch([var1, var2])
            #     with tf.GradientTape(persistent=True) as tape:
            #         tape.watch([var1, var2])
            #         # Generate psi, p from dense
            #         # psi_p = self.dense_model(batch_x)

            #         # # Generate LSTM predictions
            #         # lstm_pred = self.lstm_model(batch_x)

            #         y_pred = self(var1, training=True)
            #         psi_p = y_pred[0]
            #         psi = psi_p[:, 0]
            #         p = psi_p[:, 1]

            #         lstm_pred = y_pred[1]

            #         tape.watch(psi)
            #         psi_grads = tape.gradient(psi, var1)
            #         u = psi_grads[:, :, 1]
            #         # pdb.set_trace()
            #         # F_Model(batch_x)
            #         # loss = self.compiled_loss(batch_y, lstm_pred, regularization_losses=self.losses)

            #     print('Using tape2 now')
            #     ddu = tape2.gradient(psi_grads, var1)

            # pdb.set_trace()
            # del tape, tape2
            # var3 = tf.Variable(psi)
            # var4 = tf.Variable(psi_grads)
            # with tf.GradientTape(persistent=True) as tape3:
            #     tape3.watch([var1, var2, var3])
            #     y_pred = self(var1, training=True)
            #     psi_grads = tape.gradient(psi, var1)
            #     test_u2 = psi_grads[:, :, 1]
            # pdb.set_trace()
            with tf.GradientTape() as tape:
                y_pred = self(batch_x, training=True)
                loss = self.compiled_loss(batch_y, y_pred, regularization_losses=self.losses)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.compiled_metrics.update_state(batch_y, y_pred)
            return {m.name: m.result() for m in self.metrics}

        def call(self, inputs, training=None, mask=None):
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                psi_p = self.dense_model(inputs)
                psi = psi_p[:, 0]

            psi_grads = tape.gradient(psi, inputs)
            u = psi_grads[:, :, 0]
            v = psi_grads[:, :, 1]
            # psi_double_grads = self.find_grad(psi_grads, inputs)
            return psi
            # # test_u = tf.Tensor()
            # with tf.GradientTape(persistent=True) as tape2:
            #     tape2.watch(inputs)
            #     with tf.GradientTape(persistent=True) as tape:
            #         tape.watch(inputs)
            #         # Pass inputs to dense_model to get psi and p
            #         f = self.dense_model(inputs, training=training, mask=mask)

            #     df = tape.gradient(f, inputs)
            #     df_dx = df[:, :, 0]
            #     # psi_grads = tape.gradient(psi, inputs)
            #     # p_grads = tape.gradient(p, inputs)
            #     # u = psi_grads[:, :, 1]
            #     # v = psi_grads[:, :, 0]
            #     # p_x = p_grads[:, :, 0]
            #     # p_y = p_grads[:, :, 1]
            # d2f_dx2 = tape2.gradient(df_dx, inputs)
            # pdb.set_trace()
            # inputs = [Batch, Lookback, Inputs]
            # with tf.GradientTape(persistent=True) as tape2:
            #     tape2.watch(inputs)
            #     with tf.GradientTape(persistent=True) as tape:
            #         tape.watch(inputs)
            #         # Pass inputs to dense_model to get psi and p
            #         psi_p = self.dense_model(inputs)
            #         psi = psi_p[:, 0]
            #         p = psi_p[:, 1]

            #         # Pass inputs to LSTM to get [u, v] predictions
            #         lstm_pred = self.lstm_model(inputs)
            #         lstm_u = lstm_pred[0]
            #         lstm_v = lstm_pred[1]

            #     # First derivatives [Batch, Lookback, Input]
            #     tape2.watch(psi_p)
            #     tape2.watch(psi)
            #     tape2.watch(p)

            #     psi_grads = tape.gradient(psi, inputs)
            #     p_grads = tape.gradient(p, inputs)
            #     u = psi_grads[:, :, 1]
            #     v = psi_grads[:, :, 0]
            #     p_x = p_grads[:, :, 0]
            #     p_y = p_grads[:, :, 1]

            #     print([v.name for v in tape.watched_variables()])
            #     pdb.set_trace()
            # pdb.set_trace()
            # return [psi_p, lstm_pred]

    lstm_model = PINNS(dense_model=dense_m, lstm_model=lstm_m)
    lstm_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    # lstm_model.build(input_shape)
    # lstm_model.summary()
    # keras.utils.plot_model(lstm_model, to_file='model.png', show_shapes=True)

    # DEBUGGING
    train_generator.sciann = False
    val_generator.sciann = False
    # batch_x, batch_y = train_generator[0]
    # lstm_model.train_step(train_generator[0])

    # n_epochs = 3
    # loss_fn = keras.losses.MeanSquaredError()
    # opt = keras.optimizers.Adam()

    # best_val_loss = np.inf
    # best_val_epoch = -1

    # training_start_time = timer()
    # for epoch in range(n_epochs):
    #     epoch_start_time = timer()
    #     print(f'Epoch {epoch}/{n_epochs}')
    #     with tqdm(total=len(train_generator), unit='step') as p_bar:
    #         for step, (x_batch, y_batch) in enumerate(train_generator):
    #             start_time = timer()
    #             with tf.GradientTape() as tape:
    #                 output = lstm_model(x_batch, training=True)
    #                 loss_value = loss_fn(y_batch, output)
    #             grads = tape.gradient(loss_value, lstm_model.trainable_weights)
    #             opt.apply_gradients(zip(grads, lstm_model.trainable_weights))

    #             # Update progress
    #             p_bar.update()
    #             p_bar.postfix = f'train_loss: {loss_value:.5f}'

    #     # Validation
    #     val_loss = lstm_model.evaluate(val_generator)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_val_epoch = epoch
    #     else:
    #         print(f'Validation loss did not improve from {best_val_loss:.5f} to {val_loss:.5f}')

    #     print(f'\t** Epoch took {timer()-epoch_start_time:.3f}s | train_loss: {loss_value:.5f} | val_loss: {val_loss:.5f} **')
    # print(f'Training took {timer()-training_start_time:.3f}s | best val_loss: {best_val_loss:.5f} at epoch {best_val_epoch}')

    # %% Model training
    start_time = timer()
    print('Fitting model')
    history = lstm_model.fit(
        x=train_generator,
        epochs=2,
        validation_data=val_generator,
        verbose=1,
        # callbacks=[early_stopping, checkpoint, tensorboard],
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
    ax.set_title('Loss (Mean Squared Error)')
    ax.legend(loc='upper right')

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
