# 4.
# 循环神经网络
# RNN
# 每个RNN层有一个循环核。一个循环核有多个记忆体。
#  
# time_step不影响参数的个数。
#  
# 设
# RNN层
# 输入向量的维度
# 为input_dim
# RNN层神经元个数
# 为
# units
#  
# 则RNN层的参数个数为
# i
# n
# p
# u
# t
# _
# d
# i
# m × u
# n
# i
# t
# s + + u
# n
# i
# t
# s
# 2 + u
# n
# i
# t
# s
# input\_dim×units + +units ^ 2 + unitsinput_dim×units + +units
# 2
# +units。输出层的参数数量计算方法还是常规思路。
#  
# 为了更直观，特在下图示例中标出。 
# 以输入数据维度为5，记忆体个数为3，输出数据维度为5为例。神经网络包含一个隐藏层和一个输出层。
#  
#
#
# 代码如下：
from keras.models import Sequential
from keras import layers

time_step = 10  # time_step不影响参数数量
input_dim = 5
units = 3  # RNN层的神经元个数，也是记忆体的个数
output_dim = 5

model4 = Sequential()
# RNN层 5个神经元 输入数据维度为5
model4.add(layers.SimpleRNN(units=units, input_shape=(time_step, input_dim), activation='relu'))
# 输出层 一个神经元 输出数据维度为5
model4.add(layers.Dense(output_dim))

model4.summary()
