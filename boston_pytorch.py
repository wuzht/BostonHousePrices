import torch
import torch.nn as nn
import numpy as np

from utils import load_data, denormalize, show_loss_curve


class NeuralNetwork(nn.Module):
    def __init__(self, in_dims):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dims, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


class BostonHousingPricesDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        train_data, test_data, self.maximums, self.minimums = load_data()
        self.data = train_data if train else test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][:-1]
        y = self.data[index][-1:]
        return torch.Tensor(x), torch.Tensor(y)


class BostonHousingPricesNeuralNetwork():
    def __init__(self):
        _, _, self.maximums, self.minimums = load_data()
        self.maximums, self.minimums = torch.Tensor(self.maximums), torch.Tensor(self.minimums)
        self.model = NeuralNetwork(13)
        self.train_data = BostonHousingPricesDataset(train=True)
        self.test_data = BostonHousingPricesDataset(train=False)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=32,
            num_workers=8,
            shuffle=True
        )
        
        self.criterion = nn.MSELoss(reduce='mean')
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=5e-4
        ) if False else torch.optim.Adam(
            params=self.model.parameters(),
            lr=1e-3
        )
        self.losses = self.train()
        show_loss_curve(self.losses)

    def train(self, num_epoches=100):
        self.model.train()
        losses = []
        for epoch_id in range(num_epoches):
            for _, (x, y) in enumerate(self.train_loader):
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            train_mae = self.val(train=True, verbose=False)
            test_mae = self.val(train=False, verbose=False)
            print('[Epoch {:3d}] loss = {:.4f}, train_mae = {:.3f}, test_mae = {:.3f}'.format(epoch_id, losses[-1], train_mae, test_mae))
        return losses

    def val(self, train=True, verbose=False):
        data = self.train_data if train else self.test_data
        self.model.eval()
        with torch.no_grad():
            mae = 0
            for i, (x, y) in enumerate(data):
                y_pred = self.model(x)
                y_denorm = denormalize(y, self.minimums[-1], self.maximums[-1]).item()
                y_pred_denorm = denormalize(y_pred, self.minimums[-1], self.maximums[-1]).item()
                if verbose:
                    print('[{:3d}] y, y_pred = {:6.3f} {:6.3f}; y, y_pred = {:6.3f} {:6.3f}'.format(
                        i, y.item(), y_pred.item(), y_denorm, y_pred_denorm))
                mae += abs(y_denorm - y_pred_denorm)
            mae /= len(data)
            if verbose:
                print('MAE = {}'.format(mae))
            return mae


if __name__ == "__main__":
    pass
