import json
import pickle
import random
import time

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from torch import nn

THRES_VALID_NO_DESC = 30


class Trainer():
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = [None] * 6
        self.gru_model = None
        self.args = args

    @staticmethod
    def shuffle_list(list, fixed_seed=1):
        random.seed(fixed_seed)
        random.shuffle(list)
        return list

    def load_dataset(self, path):
        with open(path, 'rb') as f:
            raw_data = pickle.load(f)
        raw_data = Trainer.shuffle_list(raw_data)

        X = np.array(np.stack([x for x, y in raw_data]), dtype=np.float32)
        Y = np.array(np.stack([y for x, y in raw_data]), dtype=np.int32)

        sample_len = len(raw_data)
        train_split = int(sample_len * 0.8)
        valid_split = int(sample_len * 0.9)
        self.X_train, self.Y_train = X[:train_split], Y[:train_split]
        self.X_valid, self.Y_valid = X[train_split:valid_split], Y[train_split:valid_split]
        self.X_test, self.Y_test = X[valid_split:], Y[valid_split:]

    @staticmethod
    def time_string():
        return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))

    @staticmethod
    def print_info(info):
        print(Trainer.time_string(), end='\t')
        print('epoch: {}'.format(info['epoch']), end='\t')
        print('\t'.join(['{}: {:.6f}'.format(k, v) for k, v in info.items() if k != 'epoch']))

    def train(self, epoch_num, batch_size):
        lr = self.args['lr']
        weight_decay = self.args['weight_decay']
        output_path = self.args['output_path']

        assert self.gru_model is not None

        train_dataset = Data.TensorDataset(torch.Tensor(self.X_train),
                                           torch.Tensor(self.Y_train))

        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = Data.TensorDataset(torch.Tensor(self.X_valid),
                                           torch.Tensor(self.Y_valid))
        valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=512, shuffle=False)

        model = self.gru_model.to(self.device)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_func = nn.BCELoss(reduction='mean')

        history = []
        best_loss = None

        ckp_path = '{}.best_checkpoint'.format(output_path)

        valid_no_desc_cnt = 0

        for epoch in range(epoch_num):
            train_loss = 0.0

            for step, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = model(x).view(-1)
                loss = loss_func(output, y)
                train_loss += float(loss) * x.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= self.X_train.shape[0]

            Y_valid_prob = []
            valid_loss = 0
            for step, (x, y) in enumerate(valid_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = model(x).view(-1)
                valid_loss += float(loss_func(output, y)) * x.shape[0]
                Y_valid_prob.extend(output.cpu().data.numpy())

            Y_valid_prob = np.array(Y_valid_prob)
            Y_valid_pred = np.round(Y_valid_prob)
            valid_loss /= self.X_valid.shape[0]

            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                }, ckp_path)
                valid_no_desc_cnt = 0
            else:
                valid_no_desc_cnt += 1

            prfs = precision_recall_fscore_support(self.Y_valid, Y_valid_pred)
            precision, recall, f1 = prfs[0][1], prfs[1][1], prfs[2][1]
            auc = roc_auc_score(self.Y_valid, Y_valid_prob)

            info = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss,
                    'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
            self.print_info(info)
            history.append(info)

            if valid_no_desc_cnt > THRES_VALID_NO_DESC:
                break

        train_output_path = '{}.train_details'.format(output_path)
        with open(train_output_path, 'w') as f:
            json.dump(history, f, indent=2)

    def test(self):
        output_path = self.args['output_path']

        assert self.gru_model is not None

        test_dataset = Data.TensorDataset(torch.Tensor(self.X_test),
                                          torch.Tensor(self.Y_test))
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

        model = self.gru_model.to(self.device)
        loss_func = nn.BCELoss(reduction='mean')

        ckp_path = '{}.best_checkpoint'.format(output_path)
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)

        Y_test_prob = []
        test_loss = 0
        for step, (x, y) in enumerate(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            output = model(x).view(-1)
            test_loss += float(loss_func(output, y)) * x.shape[0]
            Y_test_prob.extend(output.cpu().data.numpy())

        Y_test_pred = np.round(Y_test_prob)
        test_loss /= self.X_test.shape[0]

        prfs = precision_recall_fscore_support(self.Y_test, Y_test_pred)
        precision, recall, f1 = prfs[0][1], prfs[1][1], prfs[2][1]
        auc = roc_auc_score(self.Y_test, Y_test_prob)

        info = {'test_loss': test_loss, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
                'args': self.args}
        test_output_path = '{}.test_result'.format(output_path)
        with open(test_output_path, 'w') as f:
            json.dump(info, f, indent=2)
