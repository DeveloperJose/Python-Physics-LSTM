import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from custom_lstms import LSTMState, script_lstm, double_flatten_states, flatten_states

# [Psi, P]
N_OUTPUTS = 2

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
    def __init__(self, seq_len, batch_size, n_inputs, n_lstm_layers, lstm_activations, lstm_td_activations):
        super(PINNS, self).__init__()

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_inputs = n_inputs

        self.n_lstm_layers = n_lstm_layers
        self.lstm_activations = lstm_activations
        self.lstm_td_activations = lstm_td_activations

        # Other parameters
        self.dropout = nn.Dropout(p=0)
        self.loss_fn = nn.MSELoss()

        # LSTM Branch
        self.bidirectional = False
        # self.lstm_test = nn.LSTM(input_size=n_inputs, hidden_size=lstm_activations, num_layers=n_lstm_layers, batch_first=False, bidirectional=self.bidirectional)
        self.lstm = script_lstm(input_size=n_inputs, hidden_size=lstm_activations, num_layers=n_lstm_layers, batch_first=False, bidirectional=self.bidirectional)

        if self.bidirectional:
            # Cat
            self.lstm_td = TimeDistributed(nn.Linear(lstm_activations*2, lstm_td_activations), batch_first=False)
            self.lstm_fc1 = nn.Linear(seq_len, 20)

            # Non-Cat
            # self.lstm_td = TimeDistributed(nn.Linear(lstm_activations*2, lstm_td_activations), batch_first=True)
            # self.lstm_fc1 = nn.Linear(seq_len*lstm_td_activations, 20)

            # LSTM Output
            self.lstm_fc2 = nn.Linear(20, N_OUTPUTS)
        else:
            self.lstm_td = TimeDistributed(nn.Linear(lstm_activations*2, lstm_td_activations), batch_first=False)
            self.lstm_fc1 = nn.Linear(seq_len, 20)
            self.lstm_fc2 = nn.Linear(20, N_OUTPUTS)

        # PINNs Branch
        # if use_pinns:
        #     self.dense_branch = nn.ModuleList([nn.Linear(n_inputs, dense_activations)])
        #     for i in range(n_dense_layers):
        #         self.dense_branch.append(nn.Linear(dense_activations, dense_activations*2))
        #         dense_activations *= 2
        #     self.dense_branch.append(nn.Linear(dense_activations, N_DENSE_OUTPUT))

        self.lambda1 = 1.0
        self.lambda2 = nn.Parameter(torch.tensor([1.0], requires_grad=True).cuda())

    def losses(self, input, y_true):
        # Note: input is batch first in pde_grads unlike forward()
        # Require gradients for input so we can use it in autograd
        input.requires_grad = True

        # Perform forward pass then get the gradients needed for the losses
        psi_p = self.forward(input)
        pred_u, pred_v, f_u, f_v = self.pde_grads(input, psi_p)

        return [
            self.loss_fn(pred_u, y_true[:, 0]), 
            # self.loss_fn(pred_v, y_true[:, 1]),
            # self.loss_fn(f_u, torch.zeros_like(f_u)),
            # self.loss_fn(f_v, torch.zeros_like(f_v))
        ]
        
    # def forward_pinns(self, input):
    #     dense_input = input
    #     for layer in self.dense_branch[:-1]:
    #         dense_input = torch.tanh(layer(dense_input))
    #     dense_output = self.dense_branch[-1](dense_input)
    #     return dense_output

    def init_hidden(self):
        states = [LSTMState(torch.randn(self.batch_size, self.lstm_activations).cuda(),
                torch.randn(self.batch_size, self.lstm_activations).cuda())
                for _ in range(self.n_lstm_layers)]
        return states

    def forward(self, input):
        # %% Change from batch first to timestep first for LSTM
        if input.shape[0] != self.seq_len:
            input = input.permute(1, 0, 2)
        
        # %% LSTM Pass (u_l, v_l)
        # h_0 = torch.zeros(self.n_lstm_layers*2, batch_size, self.lstm_activations, requires_grad=True).cuda()
        # c_0 = torch.zeros(self.n_lstm_layers*2, batch_size, self.lstm_activations, requires_grad=True).cuda()
        # output, (hn, cn) = self.lstm(input, (h_0, c_0))

        if self.bidirectional:
            states = self.init_hidden()
            output, out_states = self.lstm(input, states)
            hn, hc = double_flatten_states(out_states)
            
            # Cat experiment
            cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
            td_output = self.lstm_td(cat).view(self.batch_size, -1)

            # Non-Cat
            # td_output = self.lstm_td(output).view(batch_size, -1)

            # Rest
            fc1_output = self.lstm_fc1(td_output)
            fc1_relu = torch.relu(fc1_output)
            lstm_d1 = self.dropout(fc1_relu)
            lstm_output = self.lstm_fc2(lstm_d1)
        else:
            states = self.init_hidden()
            output, out_states = self.lstm(input, states)
            hn, hc = flatten_states(out_states)
            
            # Cat experiment
            cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
            td_output = self.lstm_td(cat).view(self.batch_size, -1)
            # td_output = self.lstm_td(hn).view(self.batch_size, -1)

            # Non-Cat
            # td_output = self.lstm_td(output).view(batch_size, -1)

            # Rest
            fc1_output = self.lstm_fc1(td_output)
            fc1_relu = torch.relu(fc1_output)
            lstm_d1 = self.dropout(fc1_relu)
            lstm_output = self.lstm_fc2(lstm_d1)
            
        # Use the hidden state of the last timestep OR use pooling (avg/max)
        # td_output = self.td(hn).view(-1, self.num_layers*self.lookback)
        # lstm_output = self.fc_lstm(output[:, -1, :])
        # out = self.fc(torch.mean(hn, 0))
        # out = self.fc(torch.max(hn)[0])
        return lstm_output

    def pde_grads(self, input, psi_p):
        # Note: input is batch first in pde_grads unlike forward()
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        # %% Dense Pass (psi_p)
        psi = psi_p[:, 0]
        p = psi_p[:, 1]

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
        return u, v, f_u, f_v