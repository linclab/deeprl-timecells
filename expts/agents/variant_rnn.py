import torch
import torch.nn as nn
import math

# GRUCell variant with ReLU instead of tanh
class GRUCellVariant(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCellVariant, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        update_gate, reset_gate, new_gate = torch.split(torch.matmul(input, self.weight_ih.t()) + self.bias_ih
                                                        + torch.matmul(hx, self.weight_hh.t()) + self.bias_hh,
                                                        self.hidden_size, dim=1)

        update_gate = torch.relu(update_gate)
        reset_gate = torch.relu(reset_gate)
        new_gate = torch.relu(new_gate)

        reset_hidden = torch.mul(reset_gate, hx)
        concat_input = torch.cat((input, reset_hidden), dim=1)

        new_hidden = torch.relu(torch.matmul(concat_input, self.weight_hh[:2 * self.hidden_size, :].t()) +
                                self.bias_hh[:2 * self.hidden_size] + new_gate)
        new_hidden = torch.tanh(new_hidden)

        hidden = torch.mul(1 - update_gate, hx) + torch.mul(update_gate, new_hidden)

        return hidden


# LSTMCell variant with ReLU instead of tanh
class LSTMCellVariant(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellVariant, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        h, c = hx

        gates = torch.matmul(input, self.weight_ih.t()) + self.bias_ih + torch.matmul(h, self.weight_hh.t()) + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(forgetgate, c) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.relu(cy))  # Apply ReLU nonlinearity to the cell state

        return hy, cy

