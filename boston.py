import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, denormalize, show_loss_curve


def val(data, model, y_min, y_max, verbose=True):
    mae = 0
    for i, x in enumerate(data):
        y = x[-1]
        y_pred = model(x[:-1])[0]
        y_denorm = denormalize(y, y_min, y_max)
        y_pred_denorm = denormalize(y_pred, y_min, y_max)
        if verbose:
            print('[{:3d}] y, y_pred = {:6.3f} {:6.3f}; y, y_pred = {:6.3f} {:6.3f}'.format(
                i, y, y_pred, y_denorm, y_pred_denorm))
        mae += abs(y_denorm - y_pred_denorm)
    mae /= len(data)
    if verbose:
        print('MAE = {}'.format(mae))
    return mae


class LinearNetwork(object):
    def __init__(self, num_of_weights, y_min, y_max):
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        self.y_min = y_min
        self.y_max = y_max
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
                
    def train(self, training_data, test_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for _, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
            train_mae = val(training_data, self.forward, self.y_min, self.y_max, verbose=False)
            test_mae = val(test_data, self.forward, self.y_min, self.y_max, verbose=False)
            print('[Epoch {:3d}] loss = {:.4f} train_mae = {:.3f}, test_mae = {:.3f}'.format(
                epoch_id, losses[-1], train_mae, test_mae))
        return losses


class BostonHousingPricesLinearNetwork():
    def __init__(self):
        self.train_data, self.test_data, self.maximums, self.minimums = load_data()
        self.y_min = self.minimums[-1]
        self.y_max = self.maximums[-1]
        self.net = LinearNetwork(13, self.y_min, self.y_max)
        self.losses = self.net.train(self.train_data, self.test_data, num_epoches=1000, batch_size=100, eta=0.1)
        show_loss_curve(self.losses)
    
    def val(self, train=True, verbose=True):
        data = self.train_data if train else self.test_data
        val(data, self.net.forward, self.y_min, self.y_max, verbose=True)


if __name__ == "__main__":
    pass
    