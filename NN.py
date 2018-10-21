import random
import numpy as np
from util import plot, train_test_split


class NN:
    # 初始化
    def __init__(self, train_data, train_label, hidden_num, output_num):
        self.train_data = train_data
        self.train_label = train_label
        # 训练样本的数量
        self.train_example_num = len(train_data)
        # 输出层神经元的数量
        self.input_num = len(train_data[0])
        # 隐藏层神经元的数量
        self.hidden_num = hidden_num
        # 输出层神经元的数量
        self.output_num = output_num
        # 隐藏层的数据
        self.hidden_data = np.zeros([1, hidden_num])
        # 输出层的数据
        self.output_data = np.zeros([1, output_num])
        # 输入层到隐藏层的weight
        self.theta1 = np.random.uniform(-1.0, 1.0, (hidden_num, self.input_num))
        # 隐藏层到输出层的weight
        self.theta2 = np.random.uniform(-1.0, 1.0, (output_num, hidden_num))
        # 隐藏层的bias的weight
        self.hidden_bias = np.zeros([1, hidden_num])
        # 输出层的bias的weight
        self.output_bias = np.zeros([1, output_num])

        self.error = np.zeros([1, output_num])

    # sigmod激活函数
    def sigmod(self, x):
        return 1.0 / (1 + np.exp(-x))

    # sigmid的偏导数函数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 前馈网络
    def feedforward(self, data_index):
        # 隐藏层的数据
        self.hidden_data = np.zeros([1, self.hidden_num])
        # 输出层的数据
        self.output_data = np.zeros([1, self.output_num])

        # 对一条数据进行训练
        a_1 = self.train_data[data_index]
        z_1 = a_1.dot(self.theta1.T)
        z_1 = z_1 + self.hidden_bias
        self.hidden_data = self.sigmod(z_1)

        a_2 = self.hidden_data
        z_2 = a_2.dot(self.theta2.T)
        z_2 = z_2 + self.output_bias
        self.output_data = self.sigmod(z_2)

        return self.output_data

    # BP反馈网络
    def feedback(self, learning_step, data_index):
        self.error = self.train_label[data_index] - self.output_data

        # 更新隐藏层到输出层的weight和b  .reshape()用于将向量转化为矩阵
        self.theta2 += learning_step * (self.error * self.sigmoid_derivative(self.output_data)).T.dot(self.hidden_data)
        self.output_bias += learning_step * (self.error * self.sigmoid_derivative(self.output_data))

        # 更新输入层到隐藏层的weight和b
        temp = (self.error * self.sigmoid_derivative(self.output_data)).dot(self.theta2)
        self.theta1 += learning_step * (temp * self.sigmoid_derivative(self.hidden_data)).T.dot(self.train_data[data_index].reshape(1, -1))
        self.hidden_bias += learning_step * temp * self.sigmoid_derivative(self.hidden_data)

    # 所有样本训练一次
    def train_once(self, learning_step):
        for data_index in range(self.train_example_num):
            self.feedforward(data_index)
            self.feedback(learning_step, data_index)

    # 训练并保存模型参数
    def train(self, train_num, learning_step):
        for i in range(train_num):
            self.train_once(learning_step)
            print('train times：', i)

        # 保存模型参数
        theta1 = np.array(self.theta1)
        np.savetxt('theta1.txt', theta1)
        theta2 = np.array(self.theta2)
        np.savetxt('theta2.txt', theta2)
        hidden_bias = np.array(self.hidden_bias)
        np.savetxt('hidden_bias.txt', hidden_bias)
        output_bias = np.array(self.output_bias)
        np.savetxt('output_bias.txt', output_bias)

        print('模型训练完毕')

    # 载入模型参数
    def load_param(self):
        self.theta1 = np.loadtxt('theta1.txt', dtype=np.float32)
        self.theta2 = np.loadtxt('theta2.txt', dtype=np.float32)
        self.hidden_bias = np.loadtxt('hidden_bias.txt', dtype=np.float32)
        self.output_bias = np.loadtxt('output_bias.txt', dtype=np.float32)

    # 预测一条数据
    def predict_one_example(self, example_data, example_real_r, is_plot=False):
        # 隐藏层的数据
        self.hidden_data = np.zeros([1, self.hidden_num])
        # 输出层的数据
        self.output_data = np.zeros([1, self.output_num])

        # 对一条数据进行训练
        a_1 = example_data
        z_1 = a_1.dot(self.theta1.T)
        z_1 = z_1 + self.hidden_bias
        self.hidden_data = self.sigmod(z_1)

        a_2 = self.hidden_data
        z_2 = a_2.dot(self.theta2.T)
        z_2 = z_2 + self.output_bias
        self.output_data = self.sigmod(z_2)

        r = np.argmax(self.output_data)
        real_r = np.argmax(example_real_r)

        if(is_plot):
            plot(example_data, r, real_r)

        print('预测值：', r, ', 真实值：', real_r)

        return r == real_r

    # 预测全部数据并统计准确率
    def predict(self, predict_data, real_result):

        predict_success_count = 0
        for data_index in range(len(predict_data)):
            if(self.predict_one_example(predict_data[data_index], real_result[data_index])):
                predict_success_count += 1

        # 统计准确率
        success_rate = predict_success_count / len(predict_data)
        print('预测完毕，准确率为：', success_rate)
        return success_rate


if __name__ == '__main__':
    # 读入数据集
    features = np.loadtxt('features.txt', dtype=np.float32)
    labels_t = np.loadtxt('labels_t.txt', dtype=np.int32)

    # 分割数据集为训练集和测试集, 分割比例7:3(因为数据集是按顺序的，故为随机划分)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_t, 0.7)
    # 保存测试集
    #np.savetxt('X_test.txt', X_test)
    #np.savetxt('y_test.txt', y_test)
    nn = NN(X_train, y_train, 25, 10)
    #nn.train(500, 0.05)

    nn.load_param()
    X_test = np.loadtxt('X_test.txt', dtype=np.float32)
    y_test = np.loadtxt('y_test.txt', dtype=np.int32)
    nn.predict(X_test, y_test)

    predict_indexes = random.sample(range(len(X_test)), 10)
    for i in predict_indexes:
        nn.predict_one_example(X_test[i], y_test[i], is_plot=True)
