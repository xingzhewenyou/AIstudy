import numpy as np


# 激活函数：Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid的导数
def sigmoid_derivative(x):
    return x * (1 - x)


# 二元交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# 深度神经网络类
class SimpleDNN:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size_1)
        self.b1 = np.zeros((1, hidden_size_1))
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2)
        self.b2 = np.zeros((1, hidden_size_2))
        self.W3 = np.random.randn(hidden_size_2, output_size)
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, X, y, y_pred, learning_rate):
        # 反向传播
        m = y.shape[0]  # 样本数量

        # 输出层的误差
        d_z3 = (y_pred - y) / m
        d_W3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        # 第二隐藏层的误差
        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2 * sigmoid_derivative(self.a2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        # 第一隐藏层的误差
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # 更新权重和偏置
        self.W3 -= learning_rate * d_W3
        self.b3 -= learning_rate * d_b3
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            # 计算损失
            loss = binary_cross_entropy(y, y_pred)
            # 反向传播
            self.backward(X, y, y_pred, learning_rate)
            # 打印损失
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')


# 生成数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 100个样本，每个样本有2个特征
y = np.array([[1 if x[0] + x[1] > 1 else 0] for x in X])  # 简单的二分类目标

# 定义并训练模型
model = SimpleDNN(input_size=2, hidden_size_1=4, hidden_size_2=4, output_size=1)
model.train(X, y, epochs=1000, learning_rate=0.1)

# 测试
y_pred = model.forward(X)
print("预测值：", np.round(y_pred[:5]))  # 打印前5个预测值
print("真实值：", y[:5])
