import matplotlib.pyplot as plt
import numpy as np
import random


# 绘图
def plot(example_data, predict_r, real_r):

    x = example_data.reshape(20, 20)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.imshow(x.transpose(), extent=[0, 100, 0, 1], aspect=100)
    ax.set_title('predicted value: {}, real value: {}'
                 .format(predict_r, real_r))

    plt.tight_layout()
    plt.show()


# 训练集和测试集分割
def train_test_split(X, y, train_ratio):
    z = np.hstack((X, y))
    np.random.shuffle(z)
    X = np.hsplit(z, np.array([X.shape[1]]))[0]
    y = np.hsplit(z, np.array([X.shape[1]]))[1]

    train_num = int(len(X) * train_ratio)

    X_train = X[0:train_num]
    X_test = X[train_num:]
    y_train = y[0:train_num]
    y_test = y[train_num:]

    return X_train, X_test, y_train, y_test
