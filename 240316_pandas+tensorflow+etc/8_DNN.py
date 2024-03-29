# https://blog.csdn.net/weixin_48964486/article/details/129413878

# 2.
# 全连接神经网络
# DNN
# 对全连接神经网络，
# 首先以一个简单的神经网络结构为例：一个中间层，一个输出层。中间层设定5个神经元，输出层设定1个神经元。
#  
#
# 全连接神经网络的每层参数的数量可以总结为，该层输入特征数据的数量（input_length）乘以该层神经元的数量，再加上该层神经元的数量。
#
# 代码示例如下

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

model1 = Sequential()
# 中间层 （或 隐藏层）
# 使用Dense()方法直接写第一个隐藏层，而不用写输入层时，要设定input_shape参数
model1.add(Dense(units=5,
                 input_shape=(10,)
                 )
           )
# 输出层 1个神经元
model1.add(Dense(1))
model1.summary()

# 其中中间层有55个参数，即输入的10个特征，乘以5个神经元的数量，加上5个神经元对应着5个偏置参数。10×5 + 5 = 55。
# 5
# 个神经元有5个输出值，即下一个Dense，即输出层的输入维度为5，而输出层神经元数量为1，且也对应着一个偏置，所以输出层的参数数量为5×1 + 1 = 6
# 个。两个层一共有61个参数。
#  
# 模型图示如下：
# plot_model(model1, show_shapes=True)


# 如果输入的是三维数据，(n, 10, 3)
# 为例，则在传入参数时，一定要注意，input_shape = (3, 10)，而不能写成(10, 3)。
#  
# 参数的个数与输入数据的维度input_dim无关（上边的数字3）。

model2 = Sequential()
model2.add(Dense(units=5,
                 input_shape=(3, 10)
                 )
           )

model2.add(Dense(1))
model2.summary()
