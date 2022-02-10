import torch
import torch.nn as nn
import torch.autograd as autograd

class PINNS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINNS, self).__init__()

        # 256-128-TimeDist-Flat-Output
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).cuda()
        output, (hn, cn) = self.lstm1(x, (h_0, c_0))

        # Use the hidden state of the last timestep
        out = self.fc(output[:, -1, :])

        # OR use pooling
        # out = self.fc(torch.mean(hn, 0))
        # out = self.fc(torch.max(hn)[0])

        return out