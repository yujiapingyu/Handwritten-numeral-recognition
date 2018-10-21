# Handwritten-numeral-recognition
3层神经网络实现手写数字识别

## 文件说明
（1）NN.py：实现了3层神经网络的前向传播和反向传播，并将其用于手写数字识别；<br>
（2）util.py：一些工具函数；<br>
（3）features.txt：5000个手写数字的特征文件，每个数字有20*20=400个特征，矩阵形状5000x400；<br>
（4）label_t.txt：对应features.txt中每行特征的标签值，矩阵形状5000x10，一行如(1 0 0 0 0 0 0 0 0 0)代表数字0<br>

