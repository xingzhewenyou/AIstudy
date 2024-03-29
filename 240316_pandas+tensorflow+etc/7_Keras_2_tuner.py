# https://tensorflow.google.cn/tutorials/keras/keras_tuner?hl=zh-cn

# 概述
# Keras Tuner 是一个库，可帮助您为 TensorFlow 程序选择最佳的超参数集。为您的机器学习 (ML) 应用选择正确的超参数集，这一过程称为超参数调节或超调。
#
# 超参数是控制训练过程和 ML 模型拓扑的变量。这些变量在训练过程中保持不变，并会直接影响 ML 程序的性能。超参数有两种类型：
#
# 模型超参数：影响模型的选择，例如隐藏层的数量和宽度
# 算法超参数：影响学习算法的速度和质量，例如随机梯度下降 (SGD) 的学习率以及 k 近邻 (KNN) 分类器的近邻数
# 在本教程中，您将使用 Keras Tuner 对图像分类应用执行超调。
#
# 设置

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
# 下载并准备数据集
# 在本教程中，您将使用 Keras Tuner 为某个对 Fashion MNIST 数据集内的服装图像进行分类的机器学习模型找到最佳超参数。
#
# 加载数据。


(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0
# 定义模型
# 构建用于超调的模型时，除了模型架构之外，还要定义超参数搜索空间。您为超调设置的模型称为超模型。
#
# 您可以通过两种方式定义超模型：
#
# 使用模型构建工具函数
# 将 Keras Tuner API 的 HyperModel 类子类化
# 您还可以将两个预定义的 HyperModel 类 HyperXception 和 HyperResNet 用于计算机视觉应用。
#
# 在本教程中，您将使用模型构建工具函数来定义图像分类模型。模型构建工具函数将返回已编译的模型，并使用您以内嵌方式定义的超参数对模型进行超调。


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=16, max_value=64, step=4)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

# 实例化调节器并执行超调
# 实例化调节器以执行超调。Keras Tuner 提供了四种调节器：RandomSearch、Hyperband、BayesianOptimization 和 Sklearn。在本教程中，您将使用 Hyperband 调节器。
#
# 要实例化 Hyperband 调节器，必须指定超模型、要优化的 objective 和要训练的最大周期数 (max_epochs)。


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=5,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
# Hyperband 调节算法使用自适应资源分配和早停法来快速收敛到高性能模型。该过程采用了体育竞技争冠模式的排除法。算法会将大量模型训练多个周期，并仅将性能最高的一半模型送入下一轮训练。Hyperband 通过计算 1 + logfactor(max_epochs) 并将其向上舍入到最接近的整数来确定要训练的模型的数量。
#
# 创建回调以在验证损失达到特定值后提前停止训练。


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# 运行超参数搜索。除了上面的回调外，搜索方法的参数也与 tf.keras.model.fit 所用参数相同。


tuner.search(img_train, label_train, epochs=5, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# 训练模型
# 使用从搜索中获得的超参数找到训练模型的最佳周期数。


# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=5, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# 重新实例化超模型并使用上面的最佳周期数对其进行训练。


hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# 要完成本教程，请在测试数据上评估超模型。


eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

# my_dir/intro_to_kt 目录中包含了在超参数搜索期间每次试验（模型配置）运行的详细日志和检查点。如果重新运行超参数搜索，Keras Tuner 将使用这些日志中记录的现有状态来继续搜索。要停用此行为，请在实例化调节器时传递一个附加的 overwrite = True 参数。
#
# 总结
# 在本教程中，您学习了如何使用 Keras Tuner 调节模型的超参数。要详细了解 Keras Tuner，请查看以下其他资源：
#
# TensorFlow 博客上的 Keras Tuner
# Keras Tuner 网站
# 另请查看 TensorBoard 中的 HParams Dashboard，以交互方式调节模型超参数。