import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F


class MyGRUModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_token, n_input, n_hid, n_layers, dropout=0.5, tie_weights=False):
        super(MyGRUModel, self).__init__()
        self.drop_rate = dropout
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, n_input)
        self.decoder = nn.Linear(n_hid, n_token)
        self.init_weights()

        self.rnn_type = 'GRU'
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.training = True

        # init RNN
        self.w_inputs = []
        self.w_hiddens = []
        self.b_inputs = []
        self.b_hiddens = []
        gate_size = 3 * n_hid
        # self._all_weights = []
        for layer in range(n_layers):
            layer_input_size = n_input if layer == 0 else n_hid
            self.w_inputs.append(Parameter(torch.Tensor(gate_size, layer_input_size)))
            self.w_hiddens.append(Parameter(torch.Tensor(gate_size, n_hid)))
            self.b_inputs.append(Parameter(torch.Tensor(gate_size)))
            self.b_hiddens.append(Parameter(torch.Tensor(gate_size)))
            layer_params = [self.w_inputs[layer], self.w_hiddens[layer], self.b_inputs[layer], self.b_hiddens[layer]]
        #     self._all_weights += layer_params
        # self._all_weights.append(self.drop)
        # self._all_weights.append(self.encoder)
        # self._all_weights.append(self.decoder)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def train(self, mode=True):
    #     r"""Sets the module in training mode.
    #
    #     This has any effect only on certain modules. See documentations of
    #     particular modules for details of their behaviors in training/evaluation
    #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #     etc.
    #
    #     Returns:
    #         Module: self
    #     """
    #     self.training = mode
    #     self.drop.train(mode)
    #     self.encoder.train(mode)
    #     self.decoder.train(mode)
    #     return self

    # def parameters(self):
    #     return self._all_weights

    # def zero_grad(self):
    #     r"""Sets gradients of all model parameters to zero."""
    #     for p in self.parameters():
    #         if p.grad is not None:
    #             p.grad.detach_()
    #             p.grad.zero_()

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        emb = self.drop(encoded)
        hidden, output = self.process_layers(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return self.w_hiddens[0].new_zeros(self.n_layers, bsz, self.n_hid)

    def process_layers(self, input, hidden):
        next_hidden = []
        for l in range(self.n_layers):
            all_output = []
            hy, output = self.process_steps(input, hidden[l], l)
            next_hidden.append(hy)
            all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if l < self.n_layers - 1:
                input = F.dropout(input, p=self.drop_rate, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.n_layers, *next_hidden[0].size())
        return next_hidden, input

    # def eval(self):
    #     r"""Sets the module in evaluation mode.
    #     """
    #     return self.train(False)
        # pass

    def process_steps(self, input, hidden, layer):
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.process_step(input[i], hidden, layer)
            output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    def process_step(self, input, hidden, layer):
        w_input = self.w_inputs[layer]
        w_hidden = self.w_hiddens[layer]
        b_input = self.b_inputs[layer]
        b_hidden = self.b_hiddens[layer]
        gi = F.linear(input, w_input, b_input)
        gh = F.linear(hidden, w_hidden, b_hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy
