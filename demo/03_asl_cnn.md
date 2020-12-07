<img src="./images/DLI_Header.png" style="width: 400px;">

# 卷积神经网络
本练习中，您将再次使用美国手语数据集训练模型。上一次我们已能对训练数据集获得很高的准确率，但模型并没有很好地泛化到验证数据集。这种无法很好地泛化到非训练数据上的行为称为*过拟合*。在本节中，我们将介绍一种流行的模型，称为[卷积神经网络](https://www.youtube.com/watch?v=x_VrgWTKkiM&vl=en)（CNN），特别适合读取图像并对其进行分类。

## 目标

在完成本节时，您将能够：
* 专门为CNN准备数据
* 创建更复杂的CNN模型，了解多种类型的模型层
* 训练CNN模型并观察其性能

## 加载和准备数据
我们可以更快地进入新主题，执行以下单元格来加载和准备用于训练的ASL数据集：


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
```

### 为卷积神经网络重构图像数据
您可能还记得，在上一个练习中，数据集中的单张图片采用包含 784 个像素的长列表格式。


```python
x_train.shape, x_test.shape
```




    ((27455, 784), (7172, 784))



采用此格式后，我们无法获得有关哪些像素彼此接近的全部信息。因此，我们无法应用卷积来检测特征。下面我们就来重构数据集，使其采用 28x28 像素格式。这将允许卷积读取每个图像并检测重要特征。

请注意，模型的第一个卷积层不仅需要知道图像的高度和宽度，还要了解颜色通道数。我们的图像为灰度图，因此只有一个通道。

这意味着我们需要将当前形状`(27455, 784)`转换为`(27455, 28, 28, 1)`。 为方便起见，对于希望保持不变的任何尺寸，我们都可以将`-1`传递给`reshape`方法，因此：


```python
# Format: reshape(Num_Images, Width, Height, Channels)
# Note: Passing in -1 as an argument keeps the same dimension
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
```


```python
x_train.shape
```




    (27455, 28, 28, 1)




```python
x_test.shape
```




    (7172, 28, 28, 1)



## 创建卷积模型
在您开始自己的深度学习之旅时，我们希望能确保您能在相应的指导下创建模型。假设您要解决的问题并非极端特例，那么很有可能别人已经创建了对您非常适用的模型。例如，您只需在 Google 上搜索一下，即可找到一组优秀的层来构建卷积模型。现在，我们将为您提供一个模型来有效解决分类的问题。

我们在讲座中会介绍许多不同的层，我们将在这里逐一解释它们。您不必为了记住它们而担心。


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

### 模型的组成部分
我们来回顾一下刚创建的模型的一些组件：

### Conv2D

这些是 2D 卷积层。较小的内核将仔细检查输入图像并检测对分类十分重要的特征。模型中的前面几层卷积将检测简单的特征，例如线条。后续的卷积层将检测更复杂的特征。我们来看一下第一个 Conv2D 层：
```Python
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same'...)
```
75 是指将要学习到的滤波器的数量。(3,3) 是指这些滤波器的大小。步长是指滤波器通过图像时的步进长度。填充是指从滤波器创建的输出图像是否与输入图像的大小匹配。

### BatchNormalization

如同对输入进行归一化一样，批量归一化可缩放隐藏层中的值以改善训练。如果愿意，您可在[此处阅读更多相关详细信息](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c)。

### MaxPool2D

最大值池化层把经过它的图像缩小至较低分辨率。这样有助于增强模型对物体平移（对象左右移动）的鲁棒性，同时提升模型的训练速度。

### Dropout

Dropout 是一种防止过拟合的技术。Dropout 随机选择一个神经元子集并在一次训练中将其关闭，使它们在该轮训练中不参与前向传播或反向传播。这有助于确保网络的鲁棒性和冗余性，使其不依赖网络中的任何区域来提供答案。

### Flatten

Flatten 接受某层的多维输出，并将其展平为一维数组。此层的输出称为特征向量，并将连接到最终分类层。

### Dense

在较早的模型中，我们已经见过密集层。我们的首个密集层（512 个单位）以特征向量作为输入，并学习到哪些特征对某个特定的分类贡献了多大的作用。第二个密集层（24 个单位）是输出预测的最终分类层。

## 模型总结

以上听起来信息量很大，但是不用担心。如果您要有效地训练卷积网模型，那么关键之处并不是要明白上述所有的内容，而是要知道它们有助于从图像中提取有用的信息用于分类任务。

现在，我们总结一下刚才创建的模型：


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 75)        750       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 75)        300       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 75)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 50)        33800     
    _________________________________________________________________
    dropout (Dropout)            (None, 14, 14, 50)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 14, 14, 50)        200       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 50)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 7, 7, 25)          11275     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 7, 7, 25)          100       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 25)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 400)               0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               205312    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 24)                12312     
    =================================================================
    Total params: 264,049
    Trainable params: 263,749
    Non-trainable params: 300
    _________________________________________________________________


## 编译模型

我们还像以前一样编译模型。


```python
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
```

同样，这里使用了默认优化器RMSProp。

## 训练模型
尽管模型架构差别很大，但训练过程看上去完全一样。运行下方训练脚本，看看准确率是否有所提高！


```python
model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test))
```

    Epoch 1/20
    858/858 [==============================] - 5s 6ms/step - loss: 0.3186 - accuracy: 0.9039 - val_loss: 0.2131 - val_accuracy: 0.9275
    Epoch 2/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0191 - accuracy: 0.9938 - val_loss: 0.2197 - val_accuracy: 0.9384
    Epoch 3/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0121 - accuracy: 0.9958 - val_loss: 0.6109 - val_accuracy: 0.8753
    Epoch 4/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0074 - accuracy: 0.9979 - val_loss: 0.2077 - val_accuracy: 0.9540
    Epoch 5/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0061 - accuracy: 0.9983 - val_loss: 0.2126 - val_accuracy: 0.9558
    Epoch 6/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0046 - accuracy: 0.9986 - val_loss: 0.3953 - val_accuracy: 0.9240
    Epoch 7/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0052 - accuracy: 0.9990 - val_loss: 0.2325 - val_accuracy: 0.9474
    Epoch 8/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0028 - accuracy: 0.9990 - val_loss: 0.2184 - val_accuracy: 0.9597
    Epoch 9/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0020 - accuracy: 0.9993 - val_loss: 0.5977 - val_accuracy: 0.9003
    Epoch 10/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.4198 - val_accuracy: 0.9364
    Epoch 11/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0016 - accuracy: 0.9996 - val_loss: 0.2619 - val_accuracy: 0.9579
    Epoch 12/20
    858/858 [==============================] - 4s 5ms/step - loss: 8.5945e-04 - accuracy: 0.9996 - val_loss: 1.5430 - val_accuracy: 0.8335
    Epoch 13/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0025 - accuracy: 0.9994 - val_loss: 0.3139 - val_accuracy: 0.9557
    Epoch 14/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0030 - accuracy: 0.9995 - val_loss: 0.4982 - val_accuracy: 0.9437
    Epoch 15/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0018 - accuracy: 0.9996 - val_loss: 0.4921 - val_accuracy: 0.9346
    Epoch 16/20
    858/858 [==============================] - 4s 5ms/step - loss: 5.5083e-04 - accuracy: 0.9998 - val_loss: 0.4264 - val_accuracy: 0.9467
    Epoch 17/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: 0.4820 - val_accuracy: 0.9387
    Epoch 18/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0017 - accuracy: 0.9996 - val_loss: 0.7151 - val_accuracy: 0.9081
    Epoch 19/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0017 - accuracy: 0.9995 - val_loss: 0.4817 - val_accuracy: 0.9314
    Epoch 20/20
    858/858 [==============================] - 4s 5ms/step - loss: 0.0015 - accuracy: 0.9997 - val_loss: 0.6761 - val_accuracy: 0.9163





    <tensorflow.python.keras.callbacks.History at 0x7f05076f5e80>



## 结果讨论
看起来大有改善！训练准确率非常高，且验证准确率也已得到提升。这是一个很棒的结果，因为我们所做的就是换了一个新模型。您可能还会看到验证准确率有所波动，这表明我们的模型的泛化能力还有改善余地。好在我们还有别的措施供我们使用，下一讲中我们继续讨论。

## 总结

在本节中，您利用了几种新的层来实现CNN，其表现比上一节中使用的简单的模型更好。希望您对使用准备好的数据创建和训练模型的整个过程更加熟悉。

### 清理显存
继续后面的内容前，请执行以下单元清理 GPU 显存。转至下一 notebook 之前需要执行此操作。


```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



## 下一步

在前面的几节中，您专注于模型的创建和训练。为了进一步提高性能，您的注意力将转移到*数据增强*，这是一组技术的集合，可以使您的模型在比原来更多更好的可用数据上进行训练。

请继续下一节：[*数据增强*](./04a_asl_augmentation.ipynb)。
