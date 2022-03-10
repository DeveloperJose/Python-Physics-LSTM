from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from lstm_pytorch import Settings

@torch.jit.script
def fused_pde(u, u_x, u_y, u_t, u_xx, u_yy, v, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y, lambda1, lambda2):
    f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy) 
    f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)
    return f_u, f_v

class PINNS(nn.Module):
    def __init__(self, S: Settings):
        super(PINNS, self).__init__()
        self.S = S

        # Other parameters
        self.dropout = nn.Dropout(p=0)
        self.loss_fn = nn.MSELoss()

        # LSTM Branch
        mult = 2 if S.BIDIRECTIONAL_LSTM else 1
        self.lstm = nn.LSTM(S.N_INPUTS, S.LSTM_ACTIVATIONS, batch_first=True, num_layers=S.N_LSTM_LAYERS, bidirectional=S.BIDIRECTIONAL_LSTM, dropout=0)
        self.lstm_td = TimeDistributed(nn.Linear(S.LSTM_ACTIVATIONS*mult, S.LSTM_TD_ACTIVATIONS))

        self.lstm_dense_layers = nn.ModuleList([nn.Linear(S.LSTM_TD_ACTIVATIONS, S.LSTM_DENSE_ACTIVATIONS)])
        for _ in range(S.LSTM_N_DENSE_LAYERS):
            self.lstm_dense_layers.append(nn.Linear(S.LSTM_DENSE_ACTIVATIONS, S.LSTM_DENSE_ACTIVATIONS))
        self.lstm_dense_layers.append(nn.Linear(S.LSTM_DENSE_ACTIVATIONS, S.N_LSTM_OUTPUT))

        self.lstm_fcout = nn.Linear(S.LSTM_TD_ACTIVATIONS, S.N_LSTM_OUTPUT)
        # self.lstm_fcout = nn.Linear(S.SEQ_LEN*S.LSTM_TD_ACTIVATIONS*mult, S.N_LSTM_OUTPUT)

        # if S.BIDIRECTIONAL_LSTM:
        #     # self.lstm_td = TimeDistributed(nn.Linear(S.LSTM_ACTIVATIONS*2, S.LSTM_TD_ACTIVATIONS), batch_first=True)
            


        #     # self.lstm_fc1 = nn.Linear(S.LSTM_TD_ACTIVATIONS, S.LSTM_DENSE_ACTIVATIONS)
        #     # self.lstm_fcout = nn.Linear(S.LSTM_TD_ACTIVATIONS, S.N_LSTM_OUTPUT)

            

        # else:
        #     # self.lstm_td = TimeDistributed(nn.Linear(S.LSTM_ACTIVATIONS, S.LSTM_TD_ACTIVATIONS), batch_first=True)
        #     # self.lstm_fc = nn.Linear(S.SEQ_LEN*S.LSTM_TD_ACTIVATIONS, S.N_LSTM_OUTPUT)
        #     raise Exception('Bidirectional LSTM only')

        # PINNs Branch
        dense_activations = S.DENSE_ACTIVATIONS
        self.dense_branch = nn.ModuleList([nn.Linear(S.N_INPUTS, dense_activations)])
        for i in range(S.N_DENSE_LAYERS):
            self.dense_branch.append(nn.Linear(dense_activations, dense_activations*2))
            dense_activations *= 2
        self.dense_branch.append(nn.Linear(dense_activations, S.N_DENSE_OUTPUT))

        # Learnable Parameters
        # self.lambda1 = nn.Parameter(torch.tensor([1.0], requires_grad=True).cuda())
        self.lambda1 = torch.tensor(1.0, requires_grad=False, device='cuda')
        self.lambda2 = nn.Parameter(torch.tensor([1.0], requires_grad=True, device='cuda'))
        self.lstm_w = torch.tensor(0.5, requires_grad=False, device='cuda') #nn.Parameter(torch.tensor([0.5], requires_grad=True, device='cuda'))

    def losses(self, input, y_true):
        l_losses = []
        if self.S.USE_PINNS and self.S.USE_LSTM:
            # Forward passes
            pinns_u, pinns_v, f_u, f_v = self.pde_grads(input)
            lstm_output = self.forward_lstm(input)
            lstm_u = lstm_output[:, 0]
            lstm_v = lstm_output[:, 1]

            # Combine outputs
            lw = torch.sigmoid(self.lstm_w)
            pw = 1 - lw

            pred_u = (lw * lstm_u) + (pw * pinns_u)
            pred_v = (lw * lstm_v) + (pw * pinns_v)

            # %% Losses
            l_losses = [
                self.loss_fn(pred_u, y_true[:, 0]), 
                self.loss_fn(pred_v, y_true[:, 1]),
                self.loss_fn(f_u, torch.zeros_like(f_u)),
                self.loss_fn(f_v, torch.zeros_like(f_v))
            ]
        elif self.S.USE_PINNS:
            pinns_u, pinns_v, f_u, f_v = self.pde_grads(input)
            l_losses = [
                self.loss_fn(pinns_u, y_true[:, 0]), 
                self.loss_fn(pinns_v, y_true[:, 1]),
                self.loss_fn(f_u, torch.zeros_like(f_u)),
                self.loss_fn(f_v, torch.zeros_like(f_v))
            ]
        else:
            lstm_output = self.forward_lstm(input)
            lstm_u = lstm_output[:, 0]
            lstm_v = lstm_output[:, 1]
            l_losses = [
                self.loss_fn(lstm_u, y_true[:, 0]), 
                self.loss_fn(lstm_v, y_true[:, 1]),
            ]
        return l_losses

        
    def forward_pinns(self, input):
        dense_input = input
        for layer in self.dense_branch[:-1]:
            dense_input = torch.tanh(layer(dense_input))
        dense_output = self.dense_branch[-1](dense_input)
        return dense_output

    def forward_lstm(self, input):
        # %% LSTM Pass (u_l, v_l)
        batch_size = input.shape[0]

        if self.S.BIDIRECTIONAL_LSTM:
            output, (hn, cn) = self.lstm(input)
            # output2, (hn2, cn2) = self.lstm2(output1)
            # Cat experiment
            cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
            td_output = self.lstm_td(cat).view(batch_size, -1)
            # lstm_output = self.lstm_fcout(td_output)

            lstm_fc = td_output
            for layer in self.lstm_dense_layers[:-1]:
                lstm_fc = self.dropout(torch.relu(layer(lstm_fc)))
            lstm_output = self.lstm_dense_layers[-1](lstm_fc)

            # Non-Cat
            # td_output = self.lstm_td(output).view(batch_size, -1)

            # Rest
            # lstm_fc1 = self.lstm_fc1(td_output)
            # lstm_d1 = self.dropout(torch.relu(lstm_fc1))
            # lstm_output = self.lstm_fcout(lstm_d1)
            # lstm_output = self.lstm_fcout(td_output)
        else:
            output1, (hn1, cn1) = self.lstm1(input)
            output2, (hn2, cn2) = self.lstm2(output1)

            
            td_output = self.lstm_td(output2)
            td_output = td_output.view(batch_size, -1)
            lstm_output = self.lstm_fcout(td_output)
            
            # Use the hidden state of the last timestep OR use pooling (avg/max)
            # td_output = self.td(hn).view(-1, self.num_layers*self.lookback)
            # lstm_output = self.fc_lstm(output[:, -1, :])
            # out = self.fc(torch.mean(hn, 0))
            # out = self.fc(torch.max(hn)[0])
        return lstm_output

    def forward(self, input):
        raise NotImplemented('Forward not used')

    def pde_grads(self, input):
        # Require gradients for input so we can use it in autograd
        input.requires_grad = True

        # Constraints for parameters
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        # %% Dense Pass (psi_p)
        dense_output = self.forward_pinns(input)
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
        f_u, f_v = fused_pde(u, u_x, u_y, u_t, u_xx, u_yy, v, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y, lambda1, lambda2)
        return u, v, f_u, f_v

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