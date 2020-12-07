<img src="./images/DLI_Header.png" style="width: 400px;">

# 数据增强
目前为止，我们已经选择了一个出色的模型架构，此架构的设计目的在于识别图像中的重要特征，因而它极大提高了模型的性能。我们的验证准确率仍落后于训练准确率，这意味着我们有点过拟合。换个角度来看，我们的模型在对验证数据集进行测试时，会因之前从未见过的内容而产生混淆。


为了增强模型在处理新数据时的鲁棒性，我们将以编程方式增加数据集的大小和差异。这称为[*数据增强*](https://link.springer.com/article/10.1186/s40537-019-0197-0)，是对很多深度学习应用都非常有用的技术。

数据的增加让模型在训练时能看到更多图像。数据差异的增加可帮助模型忽略不重要的特征，而只选择在分类时真正重要的特征。如此一来，在面对新数据时，模型有望在进行预测时更好地泛化。

## 目标

完成这一章节，您将能够：
* 增强 ASL 数据集
* 使用增强数据来训练改进的模型
* 将训练好的模型保存到磁盘，以进行部署

## 数据准备
我们目前在使用新的 notebook，因此需要重新加载和处理数据。为此，请执行以下单元。


```python
import tensorflow.keras as keras
import pandas as pd

# Load in our data from CSV files
train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")

# Separate out our target values
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Separate out our image vectors
x_train = train_df.values
x_test = test_df.values

# Turn our scalar targets into binary categories
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Normalize our image data
x_train = x_train / 255
x_test = x_test / 255

# Reshape the image data for the convolutional network
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
```

## 创建模型
我们还需再次创建模型。为此，请执行以下单元。这是和上一章节同样的模型架构。


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))
```

## 数据增强
编译模型前，我们应设置数据增强。

Keras 附带一个名为 `ImageDataGenerator` 的图像增强对象类。建议您 [查看此处的文档](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)。它接受一系列数据增强选项。本课程后面，我们会让您选择一个合适的增强策略。现在，请在下方查看我们所选的选项，然后执行此单元来创建该类的一个实例。


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False)  # Don't randomly flip images vertically
```

花点时间想想为何要水平翻转图像，而不是垂直翻转。有思路后，您可以参阅下方的文字。

我们的数据集由表示字母表的手部图片组成。如果以后要使用此模型对手部图像进行分类时，这些手是不可能上下倒置的，但他们可能是左撇子。这种特定领域的推理可以帮助您为自己的深度学习应用程序做出正确的决策。

## 将数据拟合到生成器

接下来，该数据生成器必须对训练数据集进行拟合。它还为实际执行图像数据的转换所需的统计信息进行预先计算。


```python
datagen.fit(x_train)
```

## 编译模型

创建数据生成器实例并拟合训练数据后，现在可以按照与前面的示例相同的方式来编译模型：


```python
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
```

注意，编译模型时可以选择不同的`优化器`，上面的编译使用了默认的优化器`RMSProp`。

## 用增强的数据进行训练
使用 Keras 图像数据生成器时，模型的训练略有不同：我们不将 x_train 和 y_train 数据集传送至模型中，而是传给生成器，并调用 `flow` 方法。这使得图像在传入模型以供训练之前即可实时得到增强并存储在内存中。

生成器可以提供无限量的数据，所以当我们使用它们来训练我们的模型时，我们需要明确设置我们希望每次训练运行多长时间。否则，生成器将产生无限多个增强图像提供给模型，使得该次训练可以无限期地进行下去。

我们使用名为`steps_per_epoch`的参数明确设置了每次训练要运行多长时间。因为通常`steps * batch_size = number_of_images_trained in an epoch`，所以我们在这里将步数设置为等于非增量数据集的大小除以batch_size（默认值为32）。

尝试运行以下训练脚本以查看结果！您会注意到，训练比以前花费更长的时间，这是有道理的，因为我们现在正在训练比以前更多的数据：


```python
model.fit(datagen.flow(x_train,y_train, batch_size=32), # Default batch_size is 32. We set it here for clarity.
          epochs=20,
          steps_per_epoch=len(x_train)/32, # Run same number of steps we would if we were not using a generator.
          validation_data=(x_test, y_test))
```

    Epoch 1/20
    858/857 [==============================] - 10s 12ms/step - loss: 1.0672 - accuracy: 0.6576 - val_loss: 0.3118 - val_accuracy: 0.9137
    Epoch 2/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.3037 - accuracy: 0.8960 - val_loss: 0.1645 - val_accuracy: 0.9366
    Epoch 3/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.1919 - accuracy: 0.9376 - val_loss: 0.4237 - val_accuracy: 0.8713
    Epoch 4/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.1449 - accuracy: 0.9519 - val_loss: 0.7842 - val_accuracy: 0.7861
    Epoch 5/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.1125 - accuracy: 0.9630 - val_loss: 0.4382 - val_accuracy: 0.8804
    Epoch 6/20
    858/857 [==============================] - 10s 11ms/step - loss: 0.1042 - accuracy: 0.9674 - val_loss: 0.5136 - val_accuracy: 0.8864
    Epoch 7/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0997 - accuracy: 0.9688 - val_loss: 0.0356 - val_accuracy: 0.9877
    Epoch 8/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0876 - accuracy: 0.9732 - val_loss: 0.0522 - val_accuracy: 0.9858
    Epoch 9/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0835 - accuracy: 0.9753 - val_loss: 0.0682 - val_accuracy: 0.9810
    Epoch 10/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0710 - accuracy: 0.9782 - val_loss: 0.1163 - val_accuracy: 0.9668
    Epoch 11/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0705 - accuracy: 0.9789 - val_loss: 0.0955 - val_accuracy: 0.9728
    Epoch 12/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0669 - accuracy: 0.9798 - val_loss: 0.0250 - val_accuracy: 0.9912
    Epoch 13/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0663 - accuracy: 0.9810 - val_loss: 0.2143 - val_accuracy: 0.9413
    Epoch 14/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0607 - accuracy: 0.9826 - val_loss: 0.1896 - val_accuracy: 0.9458
    Epoch 15/20
    858/857 [==============================] - 10s 11ms/step - loss: 0.0553 - accuracy: 0.9835 - val_loss: 1.0604 - val_accuracy: 0.8420
    Epoch 16/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0569 - accuracy: 0.9844 - val_loss: 0.1327 - val_accuracy: 0.9658
    Epoch 17/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0498 - accuracy: 0.9863 - val_loss: 1.1869 - val_accuracy: 0.8450
    Epoch 18/20
    858/857 [==============================] - 10s 11ms/step - loss: 0.0608 - accuracy: 0.9840 - val_loss: 0.0314 - val_accuracy: 0.9890
    Epoch 19/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0513 - accuracy: 0.9855 - val_loss: 0.0239 - val_accuracy: 0.9933
    Epoch 20/20
    858/857 [==============================] - 10s 12ms/step - loss: 0.0544 - accuracy: 0.9859 - val_loss: 0.0734 - val_accuracy: 0.9759





    <tensorflow.python.keras.callbacks.History at 0x7f283e8aa400>



## 讨论获得的结果
您会注意到验证准确率更高且始终如一，这意味着我们的模型已摆脱过拟合问题。它具有更好的泛化能力，因而能够更好地对新数据作出预测。

## 保存模型
现在我们已对模型进行了有效训练，下面我们就来实际应用它吧！让我们使用该模型对新图像进行分类，此过程称为推理。而在程序中使用训练过的模型则称为部署。第一步最好先将模型保存到磁盘上。然后，您可将模型传给应用该模型的各种环境中并加载。

通过`save`方法保存 Keras 中的模型是很容易的。我们可以使用不同的文件格式进行保存，但现在将采用默认格式。如果您愿意，可随时参阅 [这个文档](https://www.tensorflow.org/guide/keras/save_and_serialize)。在下一个 notebook 中，我们将加载模型，并用它来读取新的手语图片！


```python
model.save('asl_model')
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/resource_variable_ops.py:1817: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: asl_model/assets


## 总结

在本章节中，您使用 Keras 增强了数据集，结果是经过训练的模型，具有较少的过拟合和出色的测试图像结果。

### 清理显存
继续进行后面的内容前，请执行以下单元以清理 GPU 显存。


```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



## 下一步

现在，您已经过有效训练的模型保存到磁盘，在下一部分中，您将部署模型以对从未看到过的图像进行预测。

请继续学习下一个 notebook: [*模型部署*](04b_asl_predictions.ipynb).
