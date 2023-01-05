import numpy as np
import torch
from torch.autograd import Variable
from torch import autograd, optim, nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple


class AC_Net(nn.Module):
    """
    An actor-critic neural network class. Takes sensory inputs and generates a policy and a value estimate.
    """

    def __init__(self, input_dimensions, action_dimensions, batch_size, hidden_types, hidden_dimensions):

        """
        AC_Net(input_dimensions, action_dimensions, hidden_types=[], hidden_dimensions=[])
        Create an actor-critic network class.
        Required arguments:
        - input_dimensions (int): the dimensions of the input space
        - action_dimensions (int): the number of possible actions
        Optional arguments:
        - batch_size (int): the size of the batches (default = 4).
        - hidden_types (list of strings): the type of hidden layers to use, options are 'linear', 'lstm', 'gru'.
        If list is empty no hidden layers are used (default = []).
        - hidden_dimensions (list of ints): the dimensions of the hidden layers. Must be a list of
                                        equal length to hidden_types (default = []).
        """

        # call the super-class init
        super(AC_Net, self).__init__()

        # store the input dimensions
        self.input_d = input_dimensions

        # check input type
        assert (hidden_types[0] == 'linear' or hidden_types[0] == 'lstm' or hidden_types[0] == 'gru')
        self.input_type = 'vector'
        self.hidden_types = hidden_types

        # store the batch size
        self.batch_size = batch_size

        # check that the correct number of hidden dimensions are specified
        assert len(hidden_types) == len(hidden_dimensions)

        # check whether we're using hidden layers
        if not hidden_types:
            self.layers = [input_dimensions, action_dimensions]
            # no hidden layers, only input to output, create the actor and critic layers
            self.output = nn.ModuleList([
                nn.Linear(input_dimensions, action_dimensions),  # ACTOR
                nn.Linear(input_dimensions, 1)])  # CRITIC
        else:
            # to store a record of the last hidden states
            self.hx = []
            self.cx = []
            # create the hidden layers
            self.hidden = nn.ModuleList()
            ## for recording pre-relu linear cell activity
            self.cell_out = [] ##
            for i, htype in enumerate(hidden_types):
                # check if hidden layer type is correct
                assert htype in ['linear', 'lstm', 'gru']
                # get the input dimensions
                # first hidden layer
                if i == 0:
                    input_d = input_dimensions
                    output_d = hidden_dimensions[i]
                    if htype == 'linear':
                        self.hidden.append(nn.Linear(input_d, output_d))
                        self.cell_out.append(Variable(torch.zeros(self.batch_size, output_d))) ##
                        self.hx.append(None) ##
                        self.cx.append(None) ##
                    elif htype == 'lstm':
                        self.hidden.append(nn.LSTMCell(input_d, output_d))
                        self.cell_out.append(None) ##
                        self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                        self.cx.append(Variable(torch.zeros(self.batch_size, output_d)))
                    elif htype == 'gru':
                        self.hidden.append(nn.GRUCell(input_d, output_d))
                        self.cell_out.append(None) ##
                        self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                        self.cx.append(None)
                # second hidden layer onwards
                else:
                    input_d = hidden_dimensions[i - 1]
                    # get the output dimension
                    output_d = hidden_dimensions[i]
                    # construct the layer
                    if htype == 'linear':
                        self.hidden.append(nn.Linear(input_d, output_d))
                        self.cell_out.append(Variable(torch.zeros(self.batch_size, output_d))) ##
                        self.hx.append(None)
                        self.cx.append(None)
                    elif htype == 'lstm':
                        self.hidden.append(nn.LSTMCell(input_d, output_d))
                        self.cell_out.append(None) ##
                        self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                        self.cx.append(Variable(torch.zeros(self.batch_size, output_d)))
                    elif htype == 'gru':
                        self.hidden.append(nn.GRUCell(input_d, output_d))
                        self.cell_out.append(None) ##
                        self.hx.append(Variable(torch.zeros(self.batch_size, output_d)))
                        self.cx.append(None)
        # create the actor and critic layers
        self.layers = [input_dimensions] + hidden_dimensions + [action_dimensions]
        self.output = nn.ModuleList([
            nn.Linear(output_d, action_dimensions),  # actor
            nn.Linear(output_d, 1)  # critic
        ])
        # store the output dimensions
        self.output_d = output_d
        # to store a record of actions and rewards
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, temperature=1, lesion_idx=None):
        '''
        forward(x):
        Runs a forward pass through the network to get a policy and value.
        Required arguments:
          - x (torch.Tensor): sensory input to the network, should be of size batch x input_d
          - lesion_idx: if not None, set hx and cx of LSTM cells to 0 before passing input through LSTM layer
        '''

        # check the inputs
        assert x.shape[-1] == self.input_d

        # pass the data through each hidden layer
        for i, layer in enumerate(self.hidden):
            # run input through the layer depending on type
            if isinstance(layer, nn.Linear): ##
                self.cell_out[i] = layer(x)
                x = F.relu(self.cell_out[i])
                lin_activity = x
            elif isinstance(layer, nn.LSTMCell):
                if lesion_idx is None:
                    x, cx = layer(x, (self.hx[i], self.cx[i]))
                    self.hx[i] = x.clone()
                    self.cx[i] = cx.clone()
                else:
                    hx_copy = self.hx[i].clone().detach()
                    cx_copy = self.cx[i].clone().detach()
                    hx_copy[:,lesion_idx] = 0
                    cx_copy[:,lesion_idx] = 0
                    x, cx = layer(x, (hx_copy, cx_copy))
                    self.hx[i] = x.clone()
                    self.cx[i] = cx.clone()
            elif isinstance(layer, nn.GRUCell):
                if lesion_idx is None:
                    x = layer(x, self.hx[i])
                    self.hx[i] = x.clone()
                else:
                    hx_copy = self.hx[i].clone().detach()
                    hx_copy[:,lesion_idx] = 0
                    x = layer(x, hx_copy)
                    self.hx[i] = x.clone()
        # pass to the output layers
        policy = F.softmax(self.output[0](x) / temperature, dim=1)
        value = self.output[1](x)
        if isinstance(self.hidden[-1], nn.Linear):
            return policy, value, lin_activity
        else:
            return policy, value

    def reinit_hid(self):
        # to store a record of the last hidden states
        self.cell_out = []
        self.hx = []
        self.cx = []

        for i, layer in enumerate(self.hidden):
            if isinstance(layer, nn.Linear):
                self.cell_out.append(Variable(torch.zeros(self.batch_size, layer.out_features))) ##
                self.hx.append(None)##
                self.cx.append(None)##
            elif isinstance(layer, nn.LSTMCell):
                self.hx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)))
                self.cx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)))
                self.cell_out.append(None) ##
            elif isinstance(layer, nn.GRUCell):
                self.hx.append(Variable(torch.zeros(self.batch_size, layer.hidden_size)))
                self.cx.append(None)
                self.cell_out.append(None)##


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def select_action(model, policy_, value_):
    a = Categorical(policy_)
    action = a.sample()
    model.saved_actions.append(SavedAction(a.log_prob(action), value_))
    return action.item(), policy_.data[0], value_.item()


def discount_rwds(r, gamma):  # takes [1,1,1,1] and makes it [3.439,2.71,1.9,1]
    disc_rwds = np.zeros_like(r).astype(float)
    r_asfloat = r.astype(float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r_asfloat[t]
        disc_rwds[t] = running_add
    return disc_rwds


def finish_trial(model, discount_factor, optimizer, scheduler=None, **kwargs):
    '''
    Finishes a given training trial and backpropagates.
    '''

    # set the return to zero
    R = 0
    returns_ = discount_rwds(np.asarray(model.rewards), gamma=discount_factor)  # [1,1,1,1] into [3.439,2.71,1.9,1]
    saved_actions = model.saved_actions

    policy_losses = []
    value_losses = []

    returns_ = torch.Tensor(returns_)

    for (log_prob, value), r in zip(saved_actions, returns_):
        rpe = r - value.item()
        policy_losses.append(-log_prob * rpe)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([[r]]))).unsqueeze(-1))
        #   return policy_losses, value_losses
    optimizer.zero_grad() # clear gradient
    p_loss = (torch.cat(policy_losses).sum())
    v_loss = (torch.cat(value_losses).sum())
    total_loss = p_loss + v_loss
    total_loss.backward(retain_graph=True) # calculate gradient
    optimizer.step()  # move down gradient
    if scheduler is not None:
        scheduler.step()

    del model.rewards[:]
    del model.saved_actions[:]

    return p_loss, v_loss