import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt


def load_data(train_ratio=0.8):
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    data = read_csv('./housing.csv', header=None, delimiter=r"\s+", names=feature_names) # pandas.core.frame.DataFrame
    # print(data)
    # # Dimension of the dataset
    # print(np.shape(data))
    # # Let's summarize the data to see the distribution of data
    # print(data.describe())

    data = np.array(data)

    offset = int(data.shape[0] * train_ratio)
    train_data = data[:offset]

    # Normalization
    maximums, minimums = train_data.max(axis=0), train_data.min(axis=0)
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
    
    # Split dataset
    train_data = data[:offset]
    test_data = data[offset:]
    return train_data, test_data, maximums, minimums


def denormalize(x, _min, _max):
    return x * (_max - _min) + _min
    
def show_loss_curve(losses):
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()
    