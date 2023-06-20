import numpy as np
import torch
import torch.nn as nn


def expand_last_dim(x, target_len):
    """
    :param x: with shape (X)
    :param target_len: int
    Expand x into shape (X, target_len)
    """
    x_shape = x.shape
    return x.unsqueeze(-1).expand(*x_shape, target_len)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def SimpleLinearBlock(n_layer, input_dim, embed_dim=None, output_dim=None, act=False):
    if embed_dim is None:
        embed_dim = input_dim
    if output_dim is None:
        output_dim = embed_dim
    assert n_layer >= 2
    module_list = [nn.Linear(input_dim, embed_dim)]
    for layer_idx in range(n_layer - 2):
        module_list.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        module_list.append(nn.Linear(embed_dim, embed_dim))
    module_list.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
    module_list.append(nn.Linear(embed_dim, output_dim))
    if act:
        module_list.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
    return nn.Sequential(*module_list)


class SimpleFC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SimpleFC, self).__init__()
        self.in_channel = in_channel
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 1)), 64)
        print('SimpleFC In: {}, Out: {}, Embedding: {}'.format(in_channel, out_channel, embed_dim))

        self.embed_dim = embed_dim
        self.stack2double = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        self.double2simple = nn.Sequential(
            nn.Conv1d(in_channel * embed_dim, embed_dim * embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv1d(embed_dim * embed_dim, embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=embed_dim, num_layers=1)

        self.after_rnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, out_channel),
        )

    def forward(self, input_volt, connect_weight, connect_feature, hidden=None, random_init=True):
        """
        :param input_volt: with shape (batch_size, in_segment, trace_len)
        :param connect_weight: with shape (batch_size, in_segment)
        :param connect_feature: with shape (batch_size, num_feature)
        :param hidden: hidden memory for rnn, useful for network assembled inference
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and (connect_weight.ndim == 2 == connect_feature.ndim)
        assert self.in_channel == input_volt.shape[1] == connect_weight.shape[1] == connect_feature.shape[1]

        batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if hidden is None and random_init:
            hidden = torch.randn(1, batch_size, self.embed_dim).to(input_volt.device)

        input_volt = input_volt.transpose(1, 2).transpose(0, 1)
        connect_weight = expand_last(connect_weight, len_input).transpose(1, 2).transpose(0, 1)
        connect_feature = expand_last(connect_feature, len_input).transpose(1, 2).transpose(0, 1)
        syn_input = torch.stack((input_volt, connect_weight, connect_feature), dim=-1)
        double_output = self.stack2double(syn_input)
        #  with shape L x B x C x embed ---> L x (embed*C) x B
        double_output_reshape = reshape_fortran(
            double_output, (*double_output.shape[:-2], self.embed_dim * self.in_channel)).transpose(1, 2)
        #  with shape L x embed x B ---> L x B x embed
        simple_output = self.double2simple(double_output_reshape).transpose(1, 2)

        rnn_out, hidden = self.rnn(simple_output, hidden)

        return self.after_rnn(rnn_out).transpose(0, 1).transpose(1, 2), hidden


class GRUBig(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GRUBig, self).__init__()
        self.in_channel = in_channel
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 1)), 64)
        print('GRUBig In: {}, Out: {}, Embedding: {}'.format(in_channel, out_channel, embed_dim))
        self.embed_dim = embed_dim
        self.stack2double = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2), )
        self.double2simple = nn.Sequential(
            nn.Conv1d(in_channel * embed_dim, embed_dim * embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv1d(embed_dim * embed_dim, embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2), )
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=1)
        self.after_rnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, out_channel), )

    def forward(self, input_volt, connection_weights, init=None, random_init=False):
        """
        :param input_volt: shape (batch_size, in_segment, trace_len)
        :param connection_weights: shape (batch_size, in_segment)
        :param init: hidden memory for rnn
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and connection_weights.ndim == 2
        assert self.in_channel == input_volt.shape[1] == connection_weights.shape[1]
        batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if init is None and random_init:
            init = torch.randn(1, batch_size, self.embed_dim).to(input_volt.device)
        input_volt = input_volt.transpose(1, 2).transpose(0, 1)
        connect_weight = expand_last_dim(connection_weights, len_input).transpose(1, 2).transpose(0, 1)
        syn_input = torch.stack((input_volt, connect_weight), dim=-1)
        double_output = self.stack2double(syn_input)
        #  with shape L x B x C x embed ---> L x (embed*C) x B
        double_output_reshape = reshape_fortran(
            double_output, (*double_output.shape[:-2], self.embed_dim * self.in_channel)).transpose(1, 2)
        #  with shape L x embed x B ---> L x B x embed
        simple_output = self.double2simple(double_output_reshape).transpose(1, 2)
        rnn_out, hidden = self.rnn(simple_output, init)
        return self.after_rnn(rnn_out).transpose(0, 1).transpose(1, 2), hidden


class GRUMega(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers=2):
        super(GRUMega, self).__init__()
        self.in_channel, self.n_layers = in_channel, n_layers
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 2)), 64)
        print('GRUMega In: {}, Out: {}, Embedding: {}, N_layers:{}'.format(in_channel, out_channel, embed_dim,
                                                                           n_layers))
        self.embed_dim = embed_dim
        self.stack2double = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2), )
        self.double2simple = nn.Sequential(
            nn.Conv1d(in_channel * embed_dim, embed_dim * embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv1d(embed_dim * embed_dim, embed_dim, kernel_size=1, groups=embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2), )
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_layers)
        self.after_rnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(embed_dim, out_channel), )

    def forward(self, input_volt, connection_weights, init=None, random_init=False):
        """
        :param input_volt: shape (batch_size, in_segment, trace_len)
        :param connection_weights: shape (batch_size, in_segment)
        :param init: hidden memory for rnn
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and connection_weights.ndim == 2
        assert self.in_channel == input_volt.shape[1] == connection_weights.shape[1]
        # batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if init is None and random_init:
            init = torch.randn(self.n_layers, batch_size, self.embed_dim).to(input_volt.device)
        input_volt = input_volt.transpose(1, 2).transpose(0, 1)
        connect_weight = expand_last_dim(connection_weights, input_volt.shape[0]).transpose(1, 2).transpose(0, 1)
        syn_input = torch.stack((input_volt, connect_weight), dim=-1)
        double_output = self.stack2double(syn_input)
        #  with shape L x B x C x embed ---> L x (embed*C) x B
        double_output_reshape = reshape_fortran(
            double_output, (*double_output.shape[:-2], self.embed_dim * self.in_channel)).transpose(1, 2)
        #  with shape L x embed x B ---> L x B x embed
        simple_output = self.double2simple(double_output_reshape).transpose(1, 2)
        rnn_out, hidden = self.rnn(simple_output, init)
        return self.after_rnn(rnn_out).transpose(0, 1).transpose(1, 2), hidden


class PlainGRU(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers=2):
        super(PlainGRU, self).__init__()
        self.in_channel, self.n_layers = in_channel, n_layers
        complexity = 2
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 2)), 64)
        print('PlainGRU In: {}, Out: {}, Embedding: {}, N_layers:{}'.format(in_channel, out_channel, embed_dim,
                                                                            n_layers))
        self.embed_dim = embed_dim
        self.pre_rnn = SimpleLinearBlock(complexity, 2 * in_channel, embed_dim, act=True)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_layers, batch_first=True)
        self.after_rnn = SimpleLinearBlock(complexity, embed_dim, output_dim=out_channel)

    def forward(self, input_volt, connection_weights, init=None, random_init=False):
        """
        :param input_volt: shape (batch_size, in_segment, trace_len)
        :param connection_weights: shape (batch_size, in_segment)
        :param init: hidden memory for rnn
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and connection_weights.ndim == 2
        assert self.in_channel == input_volt.shape[1] == connection_weights.shape[1]
        batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if init is None and random_init:
            init = torch.randn(self.n_layers, batch_size, self.embed_dim).to(input_volt.device)

        connect_weight = expand_last_dim(connection_weights, len_input)
        syn_input = torch.cat((input_volt, connect_weight), dim=1).transpose(1, 2)
        pre_rnn = self.pre_rnn(syn_input)
        rnn_out, hidden = self.rnn(pre_rnn, init)
        return self.after_rnn(rnn_out).transpose(1, 2), hidden


class Weight2HiddenGRU(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers=2):
        super(Weight2HiddenGRU, self).__init__()
        self.in_channel, self.n_layers = in_channel, n_layers
        complexity = 2
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 2)), 64)
        print('Weight2HiddenGRU In: {}, Out: {}, Embedding: {}, N_layers:{}'.format(in_channel, out_channel, embed_dim,
                                                                                    n_layers))
        self.embed_dim = embed_dim
        self.hidden_generator = SimpleLinearBlock(complexity, in_channel, embed_dim * n_layers)
        self.pre_rnn = SimpleLinearBlock(complexity, in_channel, embed_dim, act=True)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_layers, batch_first=True)
        self.after_rnn = SimpleLinearBlock(complexity, embed_dim, output_dim=out_channel)

    def forward(self, input_volt, connection_weights, init=None, random_init=False):
        """
        :param input_volt: shape (batch_size, in_segment, trace_len)
        :param connection_weights: shape (batch_size, in_segment)
        :param init: hidden memory for rnn
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and connection_weights.ndim == 2
        assert self.in_channel == input_volt.shape[1] == connection_weights.shape[1]
        batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if init is None and random_init:
            init = self.hidden_generator(connection_weights).reshape(batch_size, self.n_layers,
                                                                     self.embed_dim).transpose(0, 1)
            # init = torch.randn(self.n_layers, batch_size, self.embed_dim).to(input_volt.device)
        pre_rnn = self.pre_rnn(input_volt.transpose(1, 2))
        rnn_out, hidden = self.rnn(pre_rnn, init)
        return self.after_rnn(rnn_out).transpose(1, 2), hidden


class DoubleGRU(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers=2):
        super(DoubleGRU, self).__init__()
        self.in_channel, self.n_layers = in_channel, n_layers
        complexity = 2
        embed_dim = min(int(np.exp2(np.floor(np.log2(in_channel)) + 2)), 64)
        print('DoubleGRU In: {}, Out: {}, Embedding: {}, N_layers:{}'.format(in_channel, out_channel, embed_dim,
                                                                            n_layers))
        self.embed_dim = embed_dim
        self.pre_gru1 = SimpleLinearBlock(complexity, in_channel, embed_dim, act=True)
        self.gru1 = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_layers, batch_first=True)
        self.v2s = SimpleLinearBlock(complexity, embed_dim, output_dim=in_channel, act=True)
        self.s2v = SimpleLinearBlock(complexity, in_channel, embed_dim, act=True)
        self.gru2 = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=n_layers, batch_first=True)
        self.after_rnn = SimpleLinearBlock(complexity, embed_dim, output_dim=out_channel)

    def forward(self, input_volt, connection_weights, init=None, random_init=False):
        """
        :param input_volt: shape (batch_size, in_segment, trace_len)
        :param connection_weights: shape (batch_size, in_segment)
        :param init: hidden memory for rnn
        :param random_init: initialize hidden state randomly
        """
        assert input_volt.ndim == 3 and connection_weights.ndim == 2
        assert self.in_channel == input_volt.shape[1] == connection_weights.shape[1]
        batch_size, len_input = input_volt.shape[0], input_volt.shape[-1]
        if init is None:
            init1 = None
            init2 = None
        else:
            init1, init2 = init[0], init[1]

        pre_s = self.pre_gru1(input_volt.transpose(1, 2))
        s, hidden1 = self.gru1(pre_s, init1)
        ws = self.v2s(s) * (connection_weights.unsqueeze(1))
        ws = self.s2v(ws)
        v, hidden2 = self.gru2(ws, init2)

        return self.after_rnn(v).transpose(1, 2), torch.stack([hidden1, hidden2], dim=0)
