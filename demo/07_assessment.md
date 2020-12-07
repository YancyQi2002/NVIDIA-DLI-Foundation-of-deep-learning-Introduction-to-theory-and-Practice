<img src="./images/DLI_Header.png" style="width: 400px;">

## 评估
恭喜您完成了今天的课程！希望您在此过程中学到了一些有价值的技能。现在该测试一下这些技能了。在此评估中，您将训练一种能够识别新鲜和腐烂水果的新模型。您需要使模型的验证准确率达到92％，才能通过评估，但我们鼓励您挑战更高的准确率。为此，您将使用先前练习中学到的技能，具体来说，我们建议您结合使用迁移学习、数据扩充和模型微调。训练好模型并在测试数据集上的准确率达到至少92％之后，请保存模型，然后评估其准确率。让我们开始吧！

### 数据集
在本练习中，您将训练一个模型来识别新鲜和腐烂的水果，数据集来自[Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification)。如果您有兴趣在课后自己开始一个新的项目，那么Kaggle是一个值得访问的好地方。现在您可详细查看`fruits`文件夹中的数据集结构。水果有六类：新鲜的苹果，新鲜的橙子，新鲜的香蕉，烂的苹果，烂的橙子和烂的香蕉。这意味着您的模型将需要有6个神经元的输出层才能成功进行分类 您还需要使用`categorical_crossentropy`作为损失函数来编译模型，因为我们有两个以上的类别。

<img src="./images/fruits.png" style="width: 600px;">

### 加载ImageNet预训练的基础模型
我们鼓励您从在ImageNet上预训练的模型开始。您需要用正确的权重加载模型，设置输入的形状，然后选择删除模型的最后一层。请记住，图像具有三个维度：高度和宽度以及多个颜色通道。因为这些图片是彩色的，所以会有红色，绿色和蓝色三个通道。我们已经为您填写了输入形状，请不要更改，否则评估将失败。如果您需要预训练模型的参考设置，请查看笔记本05b，您在那里最先实现的迁移学习。


```python
from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 1s 0us/step


### 冻结基础模型
接下来，我们建议您像在笔记本05b中一样冻结基础模型。这样做是为了使从ImageNet数据集中所学到的知识都不会在初始的训练中被破坏。


```python
# Freeze base model
base_model.trainable = False
```

### 向模型添加新层
现在该向预训练模型中添加新层了。您可以再次使用笔记本05b作为指导。请密切注意最后的全连接（Dense）层，并确保其具有正确数量的神经元以对不同类型的水果进行分类。


```python
# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)
```


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    vgg16 (Model)                (None, 7, 7, 512)         14714688  
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 6)                 3078      
    =================================================================
    Total params: 14,717,766
    Trainable params: 3,078
    Non-trainable params: 14,714,688
    _________________________________________________________________


### 编译模型
现在可以使用损失函数（loss）和衡量标准（metrics）选项来编译模型了。请记住，我们正在训练的模型是要解决多分类而不是二分类的问题。


```python
model.compile(loss = 'categorical_crossentropy' , metrics = ['accuracy'])
```

### 扩充数据
如果需要，请尝试扩充数据以改进数据集。请参考笔记本04a和笔记本05b中的数据扩充的示例。您也可以查看[Keras ImageDataGenerator类](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)的文档。 此步骤是可选的，但是您可能会发现，这对训练时能达到95％的准确率很有帮助。


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
)
```

### 加载数据集
现在应该加载训练和测试数据集了。您必须选择正确的文件夹以及图像的正确的`target_size`（它必须与您创建的模型的输入高度和宽度相匹配）。如果您需要参考，可以查看笔记本05b。


```python
# load and iterate training dataset
train_it = datagen.flow_from_directory('fruits/train/', 
                                       target_size=(224,224), 
                                       color_mode='rgb', 
                                       class_mode="categorical")
# load and iterate test dataset
test_it = datagen.flow_from_directory('fruits/test/', 
                                      target_size=(224,224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")
```

    Found 1182 images belonging to 6 classes.
    Found 329 images belonging to 6 classes.


### 训练模型
现在开始训练模型！将训练和测试数据集传递给`fit`函数，并设置所需的训练次数（epochs）。


```python
model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=30)
```

    /usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
      warnings.warn('This ImageDataGenerator specifies '


    Epoch 1/30
    37/36 [==============================] - 24s 639ms/step - loss: 2.3020 - accuracy: 0.5068 - val_loss: 0.9599 - val_accuracy: 0.7173
    Epoch 2/30
    37/36 [==============================] - 22s 608ms/step - loss: 0.7117 - accuracy: 0.7766 - val_loss: 0.5333 - val_accuracy: 0.8237
    Epoch 3/30
    37/36 [==============================] - 23s 609ms/step - loss: 0.3467 - accuracy: 0.8723 - val_loss: 0.2594 - val_accuracy: 0.8997
    Epoch 4/30
    37/36 [==============================] - 23s 609ms/step - loss: 0.2433 - accuracy: 0.9137 - val_loss: 0.2277 - val_accuracy: 0.9271
    Epoch 5/30
    37/36 [==============================] - 23s 610ms/step - loss: 0.1538 - accuracy: 0.9475 - val_loss: 0.2929 - val_accuracy: 0.8997
    Epoch 6/30
    37/36 [==============================] - 22s 608ms/step - loss: 0.1319 - accuracy: 0.9475 - val_loss: 0.3381 - val_accuracy: 0.8967
    Epoch 7/30
    37/36 [==============================] - 22s 608ms/step - loss: 0.0988 - accuracy: 0.9645 - val_loss: 0.2436 - val_accuracy: 0.9271
    Epoch 8/30
    37/36 [==============================] - 23s 610ms/step - loss: 0.0907 - accuracy: 0.9653 - val_loss: 0.1676 - val_accuracy: 0.9574
    Epoch 9/30
    37/36 [==============================] - 22s 608ms/step - loss: 0.0654 - accuracy: 0.9805 - val_loss: 0.1516 - val_accuracy: 0.9514
    Epoch 10/30
    37/36 [==============================] - 23s 609ms/step - loss: 0.0699 - accuracy: 0.9780 - val_loss: 0.0932 - val_accuracy: 0.9848
    Epoch 11/30
    37/36 [==============================] - 22s 600ms/step - loss: 0.0568 - accuracy: 0.9814 - val_loss: 0.1559 - val_accuracy: 0.9514
    Epoch 12/30
    37/36 [==============================] - 23s 610ms/step - loss: 0.0438 - accuracy: 0.9873 - val_loss: 0.1581 - val_accuracy: 0.9392
    Epoch 13/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0346 - accuracy: 0.9882 - val_loss: 0.1112 - val_accuracy: 0.9635
    Epoch 14/30
    37/36 [==============================] - 23s 616ms/step - loss: 0.0339 - accuracy: 0.9907 - val_loss: 0.1801 - val_accuracy: 0.9483
    Epoch 15/30
    37/36 [==============================] - 23s 608ms/step - loss: 0.0315 - accuracy: 0.9915 - val_loss: 0.1408 - val_accuracy: 0.9453
    Epoch 16/30
    37/36 [==============================] - 23s 610ms/step - loss: 0.0273 - accuracy: 0.9915 - val_loss: 0.0941 - val_accuracy: 0.9605
    Epoch 17/30
    37/36 [==============================] - 23s 608ms/step - loss: 0.0271 - accuracy: 0.9924 - val_loss: 0.1503 - val_accuracy: 0.9635
    Epoch 18/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0232 - accuracy: 0.9924 - val_loss: 0.1431 - val_accuracy: 0.9696
    Epoch 19/30
    37/36 [==============================] - 22s 605ms/step - loss: 0.0261 - accuracy: 0.9924 - val_loss: 0.1879 - val_accuracy: 0.9483
    Epoch 20/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0187 - accuracy: 0.9941 - val_loss: 0.0923 - val_accuracy: 0.9757
    Epoch 21/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0169 - accuracy: 0.9949 - val_loss: 0.1956 - val_accuracy: 0.9483
    Epoch 22/30
    37/36 [==============================] - 22s 600ms/step - loss: 0.0129 - accuracy: 0.9975 - val_loss: 0.1523 - val_accuracy: 0.9574
    Epoch 23/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0173 - accuracy: 0.9941 - val_loss: 0.1326 - val_accuracy: 0.9605
    Epoch 24/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0181 - accuracy: 0.9941 - val_loss: 0.0891 - val_accuracy: 0.9726
    Epoch 25/30
    37/36 [==============================] - 23s 609ms/step - loss: 0.0125 - accuracy: 0.9958 - val_loss: 0.1464 - val_accuracy: 0.9666
    Epoch 26/30
    37/36 [==============================] - 22s 608ms/step - loss: 0.0103 - accuracy: 0.9983 - val_loss: 0.1907 - val_accuracy: 0.9574
    Epoch 27/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0090 - accuracy: 0.9992 - val_loss: 0.1394 - val_accuracy: 0.9787
    Epoch 28/30
    37/36 [==============================] - 22s 607ms/step - loss: 0.0087 - accuracy: 0.9992 - val_loss: 0.1341 - val_accuracy: 0.9726
    Epoch 29/30
    37/36 [==============================] - 23s 609ms/step - loss: 0.0112 - accuracy: 0.9975 - val_loss: 0.0576 - val_accuracy: 0.9848
    Epoch 30/30
    37/36 [==============================] - 23s 611ms/step - loss: 0.0086 - accuracy: 0.9983 - val_loss: 0.2134 - val_accuracy: 0.9544





    <tensorflow.python.keras.callbacks.History at 0x7fda482fd080>



### 解冻模型以进行微调
如果您已经达到了92％的验证准确率，则此步是可选的。如果没有，我们建议您以很小的学习率尝试对模型进行微调。您可以再次使用笔记本05b作为参考。


```python
# Unfreeze the base model
base_model.trainable = FIXME

# Compile the model with a low learning rate
model.compile(FIXME),
              loss = FIXME , metrics = FIXME)
```


```python
model.fit(FIXME,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=FIXME)
```


### 评估模型

希望您现在拥有的模型具有92％或更高的验证准确率。如果没有，您可能需要返回并对模型进行更多的训练，或者对数据增强进行调整。

对验证精度满意后，您可以通过执行以下单元格来评估模型。`evaluate`函数将返回一个元组（tuple），其中第一个值是您的损失，第二个值是您的准确率。您需要获得0.92或更高的精度值。


```python
model.evaluate(test_it, steps=test_it.samples/test_it.batch_size)
```

    11/10 [================================] - 4s 398ms/step - loss: 0.1586 - accuracy: 0.9757



    [0.15863288938999176, 0.975683867931366]



### 执行评估

请执行以下2个代码单元来评估您的结果。

**注意：** `run_assessment` 假设您的模型是以 `model` 命名的，而且您的测试数据集的名字是`test_it`。无论出于什么原因您修改了上述名字，请在下面的单元中对`run_assessment`的参数做相应的修改。


```python
from run_assessment import run_assessment
```


```python
run_assessment(model, test_it)
```

    Evaluating model 5 times to obtain average accuracy...
    
    11/10 [================================] - 4s 397ms/step - loss: 0.1725 - accuracy: 0.9666
    11/10 [================================] - 4s 398ms/step - loss: 0.1015 - accuracy: 0.9818
    11/10 [================================] - 5s 431ms/step - loss: 0.1116 - accuracy: 0.9757
    11/10 [================================] - 4s 394ms/step - loss: 0.1133 - accuracy: 0.9787
    11/10 [================================] - 4s 397ms/step - loss: 0.1525 - accuracy: 0.9696
    
    Accuracy required to pass the assessment is 0.92 or greater.
    Your average accuracy is 0.9745.
    
    Congratulations! You passed the assessment!
    See instructions below to generate a certificate.


### 生成证书

如果您通过了评估，请返回课程页面（见下图）并单击Assess（评估）按钮，就会产生本课程的合格证书。

<img src="./images/assess_task.png" style="width: 800px;">
