import math
import torch
import torch.nn as nn
from torch.nn import Parameter
# from torch.nn import functional as F


class MyGRUModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, n_token, n_input, n_hid, n_layers, dropout=0.5, tie_weights=False):
        super(MyGRUModel, self).__init__()
        self.drop_rate = dropout
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, n_input)
        self.decoder = nn.Linear(n_hid, n_token)

        self.rnn_type = 'GRU'
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.training = True

        # init RNN
        self.w_inputs = []
        self.w_hiddens = []
        self.b_inputs = []
        # self.b_hiddens = []
        gate_size = 3 * n_hid
        self._all_weights = []
        for layer in range(n_layers):
            layer_input_size = n_input if layer == 0 else n_hid
            w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(gate_size, n_hid))
            b_ih = Parameter(torch.Tensor(gate_size))
            # b_hh = Parameter(torch.Tensor(gate_size))

            setattr(self, 'w_ih' + str(layer), w_ih)
            setattr(self, 'w_hh' + str(layer), w_hh)
            setattr(self, 'b_ih' + str(layer), b_ih)
            # setattr(self, 'b_hh' + str(layer), b_hh)

            self.w_inputs.append(w_ih)
            self.w_hiddens.append(w_hh)
            self.b_inputs.append(b_ih)
            # self.b_hiddens.append(b_hh)

        self.reset_parameters()
        self.init_weights()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.n_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        encoded = self.encoder(input)
        emb = self.drop(encoded)
        hidden, output = self.process_layers(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return next(self.parameters()).new_zeros(self.n_layers, bsz, self.n_hid)

    def process_layers(self, input, hidden):
        next_hidden = []
        for l in range(self.n_layers):
            all_output = []
            hy, output = self.process_states(input, hidden[l], l)
            next_hidden.append(hy)
            all_output.append(output)
            input = torch.cat(all_output, input.dim() - 1)
            if l < self.n_layers - 1:
                input = self.drop(input)
                # input = F.dropout(input, p=self.drop_rate, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.n_layers, *next_hidden[0].size())
        return next_hidden, input

    def process_states(self, input, hidden, layer):
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.gru_cell(input[i], hidden, layer)
            output.append(hidden)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return hidden, output

    def gru_cell(self, input, hidden, layer):
        # tmp = self.get_all_weights()
        # w_input = tmp[layer][0]
        # w_hidden = tmp[layer][1]
        # b_input = tmp[layer][2]
        # b_hidden = tmp[layer][3]

        w_input = self.w_inputs[layer]
        w_hidden = self.w_hiddens[layer]
        b_input = self.b_inputs[layer]
        # b_hidden = self.b_hiddens[layer]
        # gi = F.linear(input, w_input, b_input)
        # gh = F.linear(hidden, w_hidden, b_hidden)
        gi = torch.addmm(b_input, input, w_input.t())
        gh = hidden.matmul(w_hidden.t())
        # gh = torch.addmm(b_hidden, hidden, w_hidden.t())
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        # resetgate = F.sigmoid(i_r + h_r)
        resetgate = (i_r + h_r).sigmoid()
        # inputgate = F.sigmoid(i_i + h_i)
        updategate = (i_i + h_i).sigmoid()
        temp_hidden = (i_n + resetgate * h_n).tanh()
        hew_hidden = temp_hidden + updategate * (hidden - temp_hidden)

        return hew_hidden
