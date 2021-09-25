import math

import torch
from torch import nn


class LocgloModel(nn.Module):

    def __init__(self, args):
        super(LocgloModel, self).__init__()

        assert args['global'] or args['local']

        if args['global']:
            self.film_rnn = nn.ModuleList([nn.GRU(
                input_size=args['fea_dim'],
                hidden_size=args['film_rnn_dim'],
                num_layers=1,
                bidirectional=args['bidirect'],
                batch_first=True
            )])[0]

            args['film_rnn_output_dim'] = args['film_rnn_dim'] * 2 if args['bidirect'] else args['film_rnn_dim']

            self.w_beta = nn.Parameter(torch.Tensor(args['film_rnn_output_dim'], args['fea_dim']))
            self.b_beta = nn.Parameter(torch.Tensor(args['fea_dim']))

            self.w_theta = nn.Parameter(torch.Tensor(args['film_rnn_output_dim'], args['fea_dim']))
            self.b_theta = nn.Parameter(torch.Tensor(args['fea_dim']))

        if args['local']:
            self.rnn = nn.ModuleList([nn.GRU(
                input_size=args['fea_dim'],
                hidden_size=args['rnn_dim'],
                num_layers=1,
                bidirectional=args['bidirect'],
                batch_first=True
            )])[0]

            args['rnn_output_dim'] = args['rnn_dim'] * 2 if args['bidirect'] else args['rnn_dim']

            self.w_alpha = nn.Parameter(torch.Tensor(args['rnn_output_dim'], args['fea_dim']))
            self.b_alpha = nn.Parameter(torch.Tensor(args['fea_dim']))

        self.dense = nn.Linear(args['fea_dim'], 1)
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        if self.args['global']:
            bound = 1 / math.sqrt(self.args['film_rnn_output_dim'])

            nn.init.uniform_(self.w_beta, -bound, bound)
            nn.init.uniform_(self.b_beta, -bound, bound)
            nn.init.uniform_(self.w_theta, -bound, bound)
            nn.init.uniform_(self.b_theta, -bound, bound)

        if self.args['local']:
            bound = 1 / math.sqrt(self.args['rnn_output_dim'])

            nn.init.uniform_(self.w_alpha, -bound, bound)
            nn.init.uniform_(self.b_alpha, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        time_step = x.shape[1]

        if self.args['global']:
            q, _ = self.film_rnn(x, None)
            S = torch.mean(q, 1)
            beta = torch.matmul(S, self.w_beta) + self.b_beta
            beta = beta.view(batch_size, 1, self.args['fea_dim']).expand(batch_size, time_step, self.args['fea_dim'])
            theta = torch.matmul(S, self.w_theta) + self.b_theta
            theta = theta.view(batch_size, 1, self.args['fea_dim']).expand(batch_size, time_step, self.args['fea_dim'])

        if self.args['local']:
            if self.args['global']:
                x = torch.mul(beta, x) + theta
            h, _ = self.rnn(x, None)

            alpha = torch.tanh(torch.matmul(h, self.w_alpha) + self.b_alpha)

        if self.args['global'] and self.args['local']:
            xi = alpha + beta
        elif self.args['local']:
            xi = alpha
        else:
            xi = beta

        c = torch.sum(torch.mul(xi, x), 1)

        return torch.sigmoid(self.dense(c))
