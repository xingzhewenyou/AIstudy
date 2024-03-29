# 5.
# 长短期记忆神经网络
# LSTM
# LSTM模型的核心是三个门和一个记忆细胞，LSTM层的参数数量为相同参数RNN模型的RNN层参数数量的4倍（单层的4倍，而非整个模型参数数量的4倍）。
# 输入门，遗忘门，记忆细胞，输出门的公式依次如下：

from keras.models import Sequential
from keras.layers import Dense, LSTM

time_step = 10
input_dim = 5
units = 3  # RNN层的神经元个数，也是记忆体的个数
output_dim = 5

model5 = Sequential()
# LSTM层
model5.add(LSTM(units=units, input_shape=(time_step, input_dim), activation='relu'))
# 添加输出层
model5.add(Dense(units=output_dim, activation='softmax'))

model5.summary()
# ————————————————
#
# 版权声明：本文为博主原创文章，遵循
# CC
# 4.0
# BY - SA
# 版权协议，转载请附上原文出处链接和本声明。
#
# 原文链接：https: // blog.csdn.net / weixin_48964486 / article / details / 129413878