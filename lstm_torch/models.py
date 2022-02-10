import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class PINNS(nn.Module):
    def __init__(self, input_size, lstm_activations, lstm_layers, dense_activations, dense_layers):
        super(PINNS, self).__init__()

        # LSTM Branch
        self.hidden_size = lstm_activations
        self.num_layers = lstm_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc_lstm = nn.Linear(self.hidden_size, 2)

        # PINNs Dense Branch
        self.dense_branch = nn.ModuleList([nn.Linear(input_size, dense_activations)])
        for i in range(dense_layers):
            self.dense_branch.append(nn.Linear(dense_activations, dense_activations))
        self.dense_branch.append(nn.Linear(dense_activations, 2))

        # Learnable Parameters
        self.lambda1 = nn.Parameter(torch.tensor([0.5], requires_grad=True).cuda())
        self.lambda2 = nn.Parameter(torch.tensor([0.5], requires_grad=True).cuda())

        self.lstm_w = nn.Parameter(torch.tensor([0.5], requires_grad=True).cuda())

    def forward(self, input):
        # Require gradients for input so we can use it in autograd
        input.requires_grad = True

        # Constraints for parameters
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        lw = torch.sigmoid(self.lstm_w)
        pw = 1 - lw

        # %% Dense Pass (psi_p)
        # dense_input = torch.tensor(x, requires_grad=True).float().cuda()
        dense_input = input
        for layer in self.dense_branch[:-1]:
            dense_input = torch.tanh(layer(dense_input))
        dense_output = self.dense_branch[-1](dense_input)
        psi = dense_output[:, :, 0]
        p = dense_output[:, :, 1]

        # %% First Derivatives | [batch, timestep, input] | 0=X, 1=Y, 2=T
        psi_grads = autograd.grad(psi, input, grad_outputs=torch.ones_like(psi), retain_graph=True, create_graph=True)[0]
        p_grads = autograd.grad(p, input, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        u = psi_grads[:, -1, 1]
        v = -psi_grads[:, -1, 0]
        p_x = p_grads[:, -1, 0]
        p_y = p_grads[:, -1, 1]

        # Second Derivatives
        u_grads = autograd.grad(u, input, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_grads = autograd.grad(v, input, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        u_x = u_grads[:, -1, 0]
        u_y = u_grads[:, -1, 1]
        u_t = u_grads[:, -1, 2]

        v_x = v_grads[:, -1, 0]
        v_y = v_grads[:, -1, 1]
        v_t = v_grads[:, -1, 2]

        # %% Third Derivatives
        u_x_grads = autograd.grad(u_x, input, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_y_grads = autograd.grad(u_y, input, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        v_x_grads = autograd.grad(v_x, input, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_y_grads = autograd.grad(v_y, input, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]

        u_xx = u_x_grads[:, -1, 0]
        u_yy = u_y_grads[:, -1, 1]

        v_xx = v_x_grads[:, -1, 0]
        v_yy = v_y_grads[:, -1, 1]

        # %% PDEs
        f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy) 
        f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

        # %% LSTM Pass (u_l, v_l)
        batch_size = input.shape[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
        output, (hn, cn) = self.lstm1(input, (h_0, c_0))

        # Use the hidden state of the last timestep OR use pooling (avg/max)
        lstm_output = self.fc_lstm(output[:, -1, :])
        lstm_u = lstm_output[:, 0]
        lstm_v = lstm_output[:, 1]

        # out = self.fc(torch.mean(hn, 0))
        # out = self.fc(torch.max(hn)[0])

        # %% Final Combination
        pred_u = (lw * lstm_u) + (pw * u)
        pred_v = (lw * lstm_v) + (pw * v)
        
        return torch.stack((pred_u, pred_v, f_u, f_v), 1)