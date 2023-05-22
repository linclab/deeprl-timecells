import torch
import torch.nn as nn
import math
from torch.nn import Parameter
from torch import Tensor
from typing import List, Tuple
import numpy as np
import numbers


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


#class RNNLayer(jit.ScriptModule):
class RNNLayer(nn.Module):
    """
    A wrapper for customized RNN layers... inputs should match torch.nn.RNN
    conventions for batch_first=True
    """
    def __init__(self, cell, trunc, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)
        self.trunc = trunc


    #@jit.script_method
    def forward(self, input: Tensor=torch.tensor([]),
                internal: Tensor=torch.tensor([]),
                state: Tensor=torch.tensor([])) -> Tuple[Tensor, Tensor]:

        if input.size(0)==0:
            input = torch.zeros(internal.size(0),internal.size(1),self.cell.input_size,
                                device=self.cell.weight_hh.device)
        if state.size(0)==0:
            state = torch.zeros(1,input.size(0),self.cell.hidden_size,
                                device=self.cell.weight_hh.device)
        if internal.size(0)==0: # TODO: check this
            internal = torch.zeros(1,input.size(1),
                                   device=self.cell.weight_hh.device)

        inputs = input.unbind(1)
        internals = internal.unbind(1)
        state = (torch.squeeze(state,0),0) #To match RNN builtin
        #outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []

        for i in range(len(inputs)):
            if hasattr(self,'trunc') and np.mod(i,self.trunc)==0:
                state = (state[0].detach(),) #Truncated BPTT

            out, state = self.cell(inputs[i], internals[i], state)
            #Here: loop theta (don't change state... consider: loop above but then how not change state?)
            #Question: what to do about internals?...
            outputs += [out]

            # TODO: adjust out, state, outputs inputs, and return to match nn.RNN
        state = torch.unsqueeze(state[0],0) #To match RNN builtin
        return torch.stack(outputs,1), state


#class RNNCell(jit.ScriptModule):
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Default Initalization
        rootk = np.sqrt(1/hidden_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk-rootk)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk-rootk)
        # The layernorms provide learnable biases

        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h)
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = i_input + h_input
        hy = self.actfun(x + internal)
        return hy


#class LayerNormRNNCell(jit.ScriptModule):
class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, musig=[0,1]):
        super(LayerNormRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Default Initalization
        rootk = np.sqrt(1/hidden_size)
        self.weight_ih = Parameter(torch.rand(hidden_size, input_size)*2*rootk-rootk)
        self.weight_hh = Parameter(torch.rand(hidden_size, hidden_size)*2*rootk-rootk)
        # The layernorms provide learnable biases

        self.layernorm = LayerNorm(hidden_size,musig)
        #TODO: Add option for torch.sigmoid or torch.tanh
        self.actfun = torch.nn.ReLU()

    # TODO: with and without history (-h)
    #@jit.script_method
    def forward(self, input: Tensor, internal: Tensor, state: Tensor) -> Tensor:
        hx = state[0]
        i_input = torch.mm(input, self.weight_ih.t())
        h_input = torch.mm(hx, self.weight_hh.t())
        x = self.layernorm(i_input + h_input)
        hy = self.actfun(x + internal)
        return hy


#class LayerNorm(jit.ScriptModule):
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, musig):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.mu = musig[0]
        self.sig = musig[1]
        self.normalized_shape = normalized_shape

    #@jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)
        sigma = input.std(-1, keepdim=True, unbiased=False) + 0.0001
        return mu, sigma

    #@jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)
        return (input - mu) / sigma * self.sig + self.mu