import numpy as np 
import sciann as sn 
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import numpy as np

import scaling
import data

x = sn.RNNVariable('x', dtype='float64')
y = sn.RNNVariable('y', dtype='float64')
t = sn.RNNVariable('z', dtype='float64')

P = sn.RNNFunctional("P", [x, y, t], 8*[20], 'tanh')
Psi = sn.RNNFunctional("Psi", [x, y, t], 8*[20], 'tanh')

lambda1 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda1")
lambda2 = sn.Parameter(np.random.rand(), inputs=[x,y,t], name="lambda2")

u = sn.diff(Psi, y)
v = -sn.diff(Psi, x)

u_t = sn.diff(u, t)
u_x = sn.diff(u, x)
u_y = sn.diff(u, y)
u_xx = sn.diff(u, x, order=2)
u_yy = sn.diff(u, y, order=2)

v_t = sn.diff(v, t)
v_x = sn.diff(v, x)
v_y = sn.diff(v, y)
v_xx = sn.diff(v, x, order=2)
v_yy = sn.diff(v, y, order=2)

p_x = sn.diff(P, x)
p_y = sn.diff(P, y)

d1 = sn.Data(u)
d2 = sn.Data(v)
d3 = sn.Data(P)

c1 = sn.Tie(-p_x, u_t+lambda1*(u*u_x+v*u_y)-lambda2*(u_xx+u_yy))
c2 = sn.Tie(-p_y, v_t+lambda1*(u*v_x+v*v_y)-lambda2*(v_xx+v_yy))
c3 = sn.Data(u_x + v_y)
c4 = Psi*0.0

model = sn.SciModel(
    inputs=[x, y, t],
    targets=[d1, d2, d3, c1, c2, c3, c4],
    loss_func="mse",
)

USE_2D = False
INPUTS = ['X', 'Y']
OUTPUTS = ['Vu', 'Vv', 'P']
LOOKBACK = 1
BATCH_SIZE = 5024

SCALER_PATH = os.path.join('output', f'scaler_{INPUTS}_2D={USE_2D}.pkl')
SCALER_CREATION_DIRS = ['/home/jperez/data/sled250', '/home/jperez/data/sled255']

sc = scaling.load_or_create(SCALER_PATH, SCALER_CREATION_DIRS, INPUTS, OUTPUTS, USE_2D)

# gen = data.SledDataGenerator('/home/jperez/data/sled255', batch_size=BATCH_SIZE, lookback=LOOKBACK, shuffle=True, use_2D=USE_2D, inputs=INPUTS, outputs=OUTPUTS, scaler=sc, start=19, end=760+1)
# gen.sciann = True

ALL_X = []
ALL_Y = []
t_train = []

for timestep in tqdm(range(1, 638+1)):
    X, Y = data.read_np(os.path.join('/home/jperez/data/sled250', f'{timestep}.npy'), INPUTS, OUTPUTS, use_2D=False, scaler=sc)
    
    ALL_X.append(X)
    ALL_Y.append(Y)

    tarr = np.zeros((14184, 1))
    tarr[:] = timestep
    t_train.append(tarr)

print(len(ALL_X[0]))
ALL_X = np.array(ALL_X).reshape(-1, len(INPUTS))
ALL_Y = np.array(ALL_Y).reshape(-1, len(OUTPUTS))
t_train = np.array(t_train).reshape(-1, 1)

x_train = ALL_X[:, 0].reshape(-1, 1)
y_train = ALL_X[:, 1].reshape(-1, 1)
u_train = ALL_Y[:, 0].reshape(-1, 1)
v_train = ALL_Y[:, 1].reshape(-1, 1)
p_train = ALL_Y[:, 2].reshape(-1, 1)

input_data = [x_train, y_train, t_train]

data_d1 = u_train
data_d2 = v_train
data_d3 = p_train
data_c1 = 'zeros'
data_c2 = 'zeros'
data_c3 = 'zeros'
data_c4 = 'zeros'
target_data = [data_d1, data_d2, data_d3, data_c1, data_c2, data_c3, data_c4]

history = model.train(
    x_true=input_data,
    y_true=target_data,
    epochs=10,
    batch_size=100,
    shuffle=True,
    learning_rate=0.001,
    reduce_lr_after=100,
    stop_loss_value=1e-8,
    verbose=1
)