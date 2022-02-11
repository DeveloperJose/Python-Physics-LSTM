from enum import Enum
from unicodedata import bidirectional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# [Vu, Vv]
N_LSTM_OUTPUT = 2

# [Psi, P]
N_DENSE_OUTPUT = 2

class State(Enum):
    LSTM_ONLY = 1
    PINNS_ONLY = 2
    BOTH_BRANCHES = 3

# https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class PINNS(nn.Module):
    def __init__(self, seq_len, n_inputs, n_lstm_layers, lstm_activations, lstm_td_activations, n_dense_layers, dense_activations, state=State.BOTH_BRANCHES):
        super(PINNS, self).__init__()

        self.seq_len = seq_len
        self.n_inputs = n_inputs

        self.n_lstm_layers = n_lstm_layers
        self.lstm_activations = lstm_activations
        self.lstm_td_activations = lstm_td_activations

        self.n_dense_layers = n_dense_layers
        self.dense_activations = dense_activations

        self.state = state
        self.dropout = nn.Dropout(p=0)

        # LSTM Branch
        if state==State.LSTM_ONLY or state==State.BOTH_BRANCHES:
            self.bidirectional = True
            self.lstm = nn.LSTM(input_size=n_inputs, hidden_size=lstm_activations, num_layers=n_lstm_layers, batch_first=True, bidirectional=self.bidirectional)

            if self.bidirectional:
                # Cat
                self.lstm_td = TimeDistributed(nn.Linear(lstm_activations*2, lstm_td_activations), batch_first=True)
                self.lstm_fc1 = nn.Linear(seq_len, 20)

                # Non-Cat
                # self.lstm_td = TimeDistributed(nn.Linear(lstm_activations*2, lstm_td_activations), batch_first=True)
                # self.lstm_fc1 = nn.Linear(seq_len*lstm_td_activations, 20)

                # LSTM Output
                self.lstm_fc2 = nn.Linear(20, N_LSTM_OUTPUT)
            else:
                self.lstm_td = TimeDistributed(nn.Linear(lstm_activations, lstm_td_activations), batch_first=True)
                self.lstm_fc = nn.Linear(seq_len*lstm_td_activations, N_LSTM_OUTPUT)

        # PINNs Branch
        if state==State.PINNS_ONLY or state==State.BOTH_BRANCHES:
            self.dense_branch = nn.ModuleList([nn.Linear(n_inputs, dense_activations)])
            for i in range(n_dense_layers):
                self.dense_branch.append(nn.Linear(dense_activations, dense_activations))
            self.dense_branch.append(nn.Linear(dense_activations, N_DENSE_OUTPUT))

            # Learnable Parameters
            self.lambda1 = nn.Parameter(torch.tensor([1.0], requires_grad=True).cuda())
            self.lambda2 = nn.Parameter(torch.tensor([1.0], requires_grad=True).cuda())
            self.lstm_w = nn.Parameter(torch.tensor([0.1], requires_grad=True).cuda())

    def forward(self, input):
        if self.state==State.PINNS_ONLY or self.state==State.BOTH_BRANCHES:
            # Require gradients for input so we can use it in autograd
            input.requires_grad = True

            # Constraints for parameters
            lambda1 = self.lambda1
            lambda2 = self.lambda2
            lw = torch.sigmoid(self.lstm_w)
            pw = 1 - lw

            # %% Dense Pass (psi_p)
            dense_input = input
            for layer in self.dense_branch[:-1]:
                dense_input = torch.tanh(layer(dense_input))
            dense_output = self.dense_branch[-1](dense_input)
            psi = dense_output[:, :, 0]
            p = dense_output[:, :, 1]

            # %% First Derivatives | [batch, timestep, input] | 0=X, 1=Y, 2=T
            # create_graph needed for higher order derivatives, retain_graph not needed
            psi_grads = autograd.grad(psi, input, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
            p_grads = autograd.grad(p, input, grad_outputs=torch.ones_like(p), create_graph=True)[0]

            u = psi_grads[:, -1, 1]
            v = -psi_grads[:, -1, 0]
            p_x = p_grads[:, -1, 0]
            p_y = p_grads[:, -1, 1]

            # %% Second Derivatives
            u_grads = autograd.grad(u, input, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_grads = autograd.grad(v, input, grad_outputs=torch.ones_like(v), create_graph=True)[0]

            u_x = u_grads[:, -1, 0]
            u_y = u_grads[:, -1, 1]
            u_t = u_grads[:, -1, 2]

            v_x = v_grads[:, -1, 0]
            v_y = v_grads[:, -1, 1]
            v_t = v_grads[:, -1, 2]

            # %% Third Derivatives
            u_x_grads = autograd.grad(u_x, input, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            u_y_grads = autograd.grad(u_y, input, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
            v_x_grads = autograd.grad(v_x, input, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
            v_y_grads = autograd.grad(v_y, input, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

            u_xx = u_x_grads[:, -1, 0]
            u_yy = u_y_grads[:, -1, 1]

            v_xx = v_x_grads[:, -1, 0]
            v_yy = v_y_grads[:, -1, 1]

            # %% PDEs
            f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy) 
            f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

        if self.state==State.LSTM_ONLY or self.state==State.BOTH_BRANCHES:
            # %% LSTM Pass (u_l, v_l)
            batch_size = input.shape[0]
            # h_0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_activations, requires_grad=True).cuda()
            # c_0 = torch.zeros(self.n_lstm_layers, batch_size, self.lstm_activations, requires_grad=True).cuda()
            # output, (hn, cn) = self.lstm(input, (h_0, c_0))

            if self.bidirectional:
                output, (hn, cn) = self.lstm(input)
                
                # Cat experiment
                cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
                td_output = self.lstm_td(cat).view(batch_size, -1)

                # Non-Cat
                # td_output = self.lstm_td(output).view(batch_size, -1)

                # Rest
                lstm_d1 = self.dropout(torch.relu(self.lstm_fc1(td_output)))
                lstm_output = self.lstm_fc2(lstm_d1)
                # lstm_output = self.lstm_fc(td_output)
            else:
                output, (hn, cn) = self.lstm(input)
                td_output = self.lstm_td(output).view(batch_size, -1)
                lstm_output = self.lstm_fc(td_output)

            # Use the hidden state of the last timestep OR use pooling (avg/max)
            # td_output = self.td(hn).view(-1, self.num_layers*self.lookback)
            # lstm_output = self.fc_lstm(output[:, -1, :])
            # out = self.fc(torch.mean(hn, 0))
            # out = self.fc(torch.max(hn)[0])

        if self.state==State.BOTH_BRANCHES:
            # %% Final Combination
            lstm_u = lstm_output[:, 0]
            lstm_v = lstm_output[:, 1]
            pred_u = (lw * lstm_u) + (pw * u)
            pred_v = (lw * lstm_v) + (pw * v)
            return torch.stack((pred_u, pred_v, f_u, f_v), 1)
        elif self.state==State.PINNS_ONLY:
            return torch.stack((u, v, f_u, f_v), 1)
        else:
            return lstm_output