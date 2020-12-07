<img src="./images/DLI_Header.png" style="width: 400px;">

# 序列数据

本节中，我们将不再关注视觉数据，而是改为处理语言。与静态图像不同，语言通常是序列数据。这意味着，此类数据的顺序十分重要。序列数据的其他示例包括随时间变化的股价数据或天气数据。包含静止图像的视频同样也是序列数据。数据中的元素与其前后元素都有关系，而这一点就要求我们采用不同的方法。

## 目标

完成本节内容的学习后，您将能够：
* 准备要在递归神经网络 (RNN) 中使用的序列数据
*构建和训练模型以执行单词预测

## 标题生成器

本节中，我们将构建可以根据几个起始词预测标题的模型。您可能已在搜索栏、手机上或文本编辑器中见到过用于自动补全句子的文本预测器。很多出色的文本预测器模型都使用非常大的数据集进行训练，并且训练时需要耗费大量时间和处理能力。本练习中的预测器非常简单，但它会让您接触到语言处理、序列数据以及我们通过序列数据训练时所使用的模型类型，即*递归神经网络* (*RNN*)。

## 读入数据

我们的数据集将由[《纽约时报》](https://www.nytimes.com/)几个月来的所有文章标题组成。首先我们将读入所有文章标题。这些文章都存储在 CSV 文件中，因此正如我们在先前小节中对 CSV 数据执行的操作一样，我们将使用 Pandas 读入它们。一些标题列为“未知”，因此我们会将这些标题过滤掉：


```python
import os 
import pandas as pd

nyt_dir = 'nyt_dataset/articles/'

all_headlines = []
for filename in os.listdir(nyt_dir):
    if 'Articles' in filename:
        # Read in all the data from the CSV file
        headlines_df = pd.read_csv(nyt_dir + filename)
        
        # Add all of the headlines to our list
        all_headlines.extend(list(headlines_df.headline.values))
        
len(all_headlines)
```




    9335



下面我们就来看看前几个标题：


```python
all_headlines[:20]
```




    ['Virtual Coins, Real Resources',
     'U.S. Advances Military Plans for North Korea',
     'Mr. Trump and the ‘Very Bad Judge’',
     'To Erase Dissent, China Bans Pooh Bear and ‘N’',
     'Loans Flowed to Kushner Cos. After Visits to the White House',
     'China Envoy Intends To Ease Trade Tensions',
     'President Trump’s Contradictory, and Sometimes False, Comments About Gun Policy to Lawmakers',
     'Classic Letter Puzzle',
     'Silicon Valley Disruption In an Australian School',
     '‘The Assassination of Gianni Versace’ Episode 6: A Nothing Man',
     'Graduate',
     'Trevor Noah Is Stunned by Trump’s Turnabout on Gun Control',
     'Is ‘Black Panther’ a ‘Defining Moment’ for the United States — and Particularly for Black America?',
     'Unknown',
     'No Pension? You Can ‘Pensionize’ Your Savings',
     'Goodbye, Pay-as-You-Wish',
     'U.S. Closes Door on Christians Who Fled Iran',
     'A Gang’s Fearsome Reputation,  Further Inflated by the President',
     'Trial of Killer’s Widow:  Scared Victim of Abuse Or Cunning Accomplice?',
     'Scintillating, and Serene, São Paulo']



## 清洗数据

自然语言处理任务（计算机在其中处理语言）的一个重要部分是按照计算机可以理解的方式处理文本。稍后我们将提取数据集中出现的每个词并用数字进行表示。这是*分词*（Tokenization）过程的其中一环。

在此之前，我们需要确保我们拥有良好的数据。有些标题被列为“未知”，我们不想在训练集中使用这些标题，因此我们将其过滤掉：


```python
# Remove all headlines with the value of "Unknown"
all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)
```




    8603



让我们再看一看：


```python
all_headlines[:20]
```




    ['Virtual Coins, Real Resources',
     'U.S. Advances Military Plans for North Korea',
     'Mr. Trump and the ‘Very Bad Judge’',
     'To Erase Dissent, China Bans Pooh Bear and ‘N’',
     'Loans Flowed to Kushner Cos. After Visits to the White House',
     'China Envoy Intends To Ease Trade Tensions',
     'President Trump’s Contradictory, and Sometimes False, Comments About Gun Policy to Lawmakers',
     'Classic Letter Puzzle',
     'Silicon Valley Disruption In an Australian School',
     '‘The Assassination of Gianni Versace’ Episode 6: A Nothing Man',
     'Graduate',
     'Trevor Noah Is Stunned by Trump’s Turnabout on Gun Control',
     'Is ‘Black Panther’ a ‘Defining Moment’ for the United States — and Particularly for Black America?',
     'No Pension? You Can ‘Pensionize’ Your Savings',
     'Goodbye, Pay-as-You-Wish',
     'U.S. Closes Door on Christians Who Fled Iran',
     'A Gang’s Fearsome Reputation,  Further Inflated by the President',
     'Trial of Killer’s Widow:  Scared Victim of Abuse Or Cunning Accomplice?',
     'Scintillating, and Serene, São Paulo',
     'Can Venezuela Be Saved?']



我们还希望删除标点符号并使句子全部小写，因为这将使我们的模型更易于训练。无论以“！”或 “？” 结尾的行，还是单词大写（如“ The”）或小写（如“ the”），就我们的目的而言，他们之间几乎没有差异。如果可以使用更少的唯一标记（tokens），我们的模型将更易于训练。

我们可以在分词之前对句子进行过滤，但是我们不需要这样做，因为所有这些都可以使用Keras的Tokenizer完成。

## 分词

现在，我们的数据集包含一组标题，每个标题由一系列单词组成。我们希望给我们的模型一种可以理解的方式表示这些单词。通过分词，我们将一段文本分割成多个以空格分隔的小块（tokens，或称标记），本例中为词，然后为每个唯一的单词分配一个数字，因为模型可以通过这种方式理解数据。Keras 提供了一个叫 `Tokenizer` 的类，可以帮助我们对数据进行分词。您可以在 [这个文档](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) 中阅读到更多相关内容。

```python
tf.keras.preprocessing.text.Tokenizer(
    num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
    split=' ', char_level=False, oov_token=None, document_count=0, **kwargs
)
```

观察一下Keras中的[Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)类，我们看到它已经为我们的例子设置了默认值 。`filters`字符串已经删除了标点符号，`lower`标志则将单词设置为小写。


```python
from tensorflow.keras.preprocessing.text import Tokenizer

# Tokenize the words in our headlines
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_headlines)
total_words = len(tokenizer.word_index) + 1
print('Total words: ', total_words)
```

    Total words:  11753


我们可以快速浏览一下word_index字典，以了解标记生成器如何保存单词：


```python
# Print a subset of the word_index dictionary created by Tokenizer
subset_dict = {key: value for key, value in tokenizer.word_index.items() \
               if key in ['a','man','a','plan','a','canal','panama']}
print(subset_dict)
```

    {'a': 2, 'plan': 82, 'man': 138, 'panama': 2549, 'canal': 11469}


我们可以使用 `texts_to_sequences` 方法查看分词器是如何表示这些词的：


```python
tokenizer.texts_to_sequences(['a','man','a','plan','a','canal','panama'])
```




    [[2], [138], [2], [82], [2], [11469], [2549]]



## 创建序列

现在我们已对数据进行分词，并将每个词转变为一个代表性的数字，下面我们就来创建标题的标记序列（token sequence）。这些序列是我们训练深度学习模型时要使用的数据。

例如，我们提取此标题“nvidia launches ray tracing gpus”。每个词都将替换为一个对应的数字，例如 nvidia - 5、launches - 22、ray - 94、tracing - 16、gpus - 102。完整序列将为：[5, 22, 94, 16, 102]，但使用该标题内较短的序列（例如“nvidia launches”）对模型进行训练也是有价值的。我们将提取每个标题，并创建一组序列来填充数据集。现在，我们就使用分词器将标题转换为一组序列。


```python
# Convert data to sequence of tokens 
input_sequences = []
for line in all_headlines:
    # Convert our headline into a sequence of tokens
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    # Create a series of sequences for each headline
    for i in range(1, len(token_list)):
        partial_sequence = token_list[:i+1]
        input_sequences.append(partial_sequence)

print(tokenizer.sequences_to_texts(input_sequences[:5]))
input_sequences[:5]
```

    ['virtual coins', 'virtual coins real', 'virtual coins real resources', 'u s', 'u s advances']





    [[1616, 5242],
     [1616, 5242, 163],
     [1616, 5242, 163, 2514],
     [28, 26],
     [28, 26, 2515]]



## 填充序列

我们的序列现在长短不一。要使模型能够使用这些数据进行训练，我们需要让所有序列等长。为执行此操作，我们将对序列进行填充。Keras 提供了我们可以使用的内置 `pad_sequences` [方法](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)。


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Determine max sequence length
max_sequence_len = max([len(x) for x in input_sequences])

# Pad all sequences with zeros at the beginning to make them all max length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[0]
```




    array([   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0, 1616, 5242], dtype=int32)



## 创建 Predictors 和 Target

我们还需将每个序列分割成 predictors 和 target 两部分。序列的最后一个词即为 target，头几个词则为 predictors。以一个完整标题为例：“nvidia releases ampere graphics cards”

<table>
<tr><td>PREDICTORS </td> <td>           TARGET </td></tr>
<tr><td>nvidia                   </td> <td>  releases </td></tr>
<tr><td>nvidia releases               </td> <td>  ampere </td></tr>
<tr><td>nvidia releases ampere      </td> <td>  graphics</td></tr>
<tr><td>nvidia releases ampere graphics </td> <td>  cards</td></tr>
</table>


```python
# Predictors are every word except the last
predictors = input_sequences[:,:-1]
# Labels are the last word
labels = input_sequences[:,-1]
labels[:5]
```




    array([5242,  163, 2514,   26, 2515], dtype=int32)



与先前的练习一样，我们的目标是多分类，即对总词汇表中的所有的词进行预测，而其中输出概率最大的那个词就是我们的预测目标。我们会让网络预测二进制类别，而非标量数。


```python
from tensorflow.keras import utils

labels = utils.to_categorical(labels, num_classes=total_words)
```

## 创建模型

在我们的模型中，我们将使用几个新型的层来处理序列化的数据。

### 嵌入层

模型第一层是嵌入层：

```Python
model.add(Embedding(input_dimension, output_dimension, input_length=input_len))
```

此层将提取分词后的序列，并为训练数据集中的所有词学习嵌入向量。此层会以向量形式表示每个词，并且该向量内的信息将包含每个词之间的关系。您可[在此处](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)详细了解嵌入层。

### 长短期记忆层

下一个非常重要的层是长短期记忆层，通常称为长短期记忆网络 (LSTM)。LSTM 是一类递归神经网络 (RNN)。与目前为止我们所见过的传统前馈网络不同，递归神经网络包含一些循环，能够持久保留输入序列中的历史信息。以下是递归神经网络的表示形式：

<img src="./images/rnn_rolled.png" style="width: 150px;">

新信息 (x) 传入该网络后，网络除了给出预测 (h) 之外，还将一些信息循环回 RNN。在下一次时间步输入新数据时，RNN 利用这些信息和新数据一起来做下一个预测。这可能看起来有点复杂，所以让我们展开来看一下：

<img src="./images/rnn_unrolled.png" style="width: 600px;">

可以看到，当一条新数据 (x) 送进网络中时，该网络不仅会给出预测 (h)，还会将一些信息传递到下一层。下一层会获取另一条数据，但也要向上一层学习。

传统 RNN 遭受的问题是，较新信息的贡献大于更早之前的信息。LSTM 是一种特殊类型的递归层，能够学习和保留较早之前的信息。如要详细了解 RNN 和 LSTM，建议您参阅 [此文章](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

那么，现在让我们来创建模型：


```python
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Input is max sequence length - 1, as we've removed the last word for the label
input_len = max_sequence_len - 1 

model = Sequential()

# Add input embedding layer
model.add(Embedding(total_words, 10, input_length=input_len))

# Add LSTM layer with 100 units
model.add(LSTM(100))
model.add(Dropout(0.1))

# Add output layer
model.add(Dense(total_words, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 27, 10)            117530    
    _________________________________________________________________
    lstm (LSTM)                  (None, 100)               44400     
    _________________________________________________________________
    dropout (Dropout)            (None, 100)               0         
    _________________________________________________________________
    dense (Dense)                (None, 11753)             1187053   
    =================================================================
    Total params: 1,348,983
    Trainable params: 1,348,983
    Non-trainable params: 0
    _________________________________________________________________


## 编译模型

和以前一样，我们使用多分类交叉熵作为损失函数来编译模型，因为我们从所有的词中预测出一个单词。在这种情况下，我们将不使用准确性作为度量标准，因为文本预测不会以与图像分类相同的方式测量准确性。

我们还将选择适合 LSTM 任务的特定优化器，此优化器称为 *Adam* 优化器。优化器的细节知识稍稍超出了本课程的范围，但您必须要知道，有些优化器在处理不同的深度学习任务时可能会更加出色。您可[在此处](https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3)详细了解这些优化器，包括Adam优化器。


```python
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

## 训练模型

与前面的部分相似，我们使用相同的调用来训练模型。我们将训练30次，这将需要几分钟。您会注意到，在模型编译未设置metrics的情况下，我们没有在模型训练期间看到训练集上的准确率或者验证集上的准确性。


```python
model.fit(predictors, labels, epochs=30, verbose=1)
```

    Epoch 1/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 7.8924
    Epoch 2/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 7.4747
    Epoch 3/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 7.2730
    Epoch 4/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 7.0325
    Epoch 5/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 6.7829
    Epoch 6/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 6.5247
    Epoch 7/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 6.2718
    Epoch 8/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 6.0281
    Epoch 9/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 5.7891
    Epoch 10/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 5.5590
    Epoch 11/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 5.3390
    Epoch 12/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 5.1297
    Epoch 13/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.9307
    Epoch 14/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.7459
    Epoch 15/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.5740
    Epoch 16/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.4076
    Epoch 17/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.2513
    Epoch 18/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 4.1089
    Epoch 19/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.9685
    Epoch 20/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.8493
    Epoch 21/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.7241
    Epoch 22/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.6160
    Epoch 23/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.5103
    Epoch 24/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.4117
    Epoch 25/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.3218
    Epoch 26/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.2296
    Epoch 27/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.1517
    Epoch 28/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 3.0712
    Epoch 29/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 2.9959
    Epoch 30/30
    1666/1666 [==============================] - 9s 5ms/step - loss: 2.9286





    <tensorflow.python.keras.callbacks.History at 0x7ffa6aaf9320>



## 讨论结果

可以看到，损失已经随着训练的进行而减少。我们可以进一步训练模型以减少损失，但这需要花费一些时间，况且我们现在也不必寻找完美的文本预测器。接下来，就让我们尝试使用模型进行预测。

## 进行预测

要进行预测，我们需要提取种子文本，并按照准备数据集的方式来准备这类文本。这表示我们需对这类文本进行分词和填充。完成此操作后，我们可以将其传入模型，以供模型作出预测。我们将创建一个称为`predict_next_token`的函数来执行此操作：


```python
def predict_next_token(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict_classes(token_list, verbose=0)
    return prediction
```


```python
prediction = predict_next_token("today in new york")
prediction
```

    WARNING:tensorflow:From <ipython-input-16-bd91571ab9e2>:4: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    Instructions for updating:
    Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).





    array([377])



下面我们就来使用分词器对预测出的词进行解码：


```python
tokenizer.sequences_to_texts([prediction])
```




    ['country']



## 生成新标题

现在我们已能预测新词，下面我们就来创建一个能够预测包含多个词的标题的函数。填入下方函数以创建任意长度的新标题。如果需要帮助，可单击下方的三个点来显示解决方案。


```python
def generate_headline(seed_text, next_words=1):
    for _ in range(next_words):
        # Predict next token
        prediction = predict_next_token(seed_text)
        # Convert token to word
        next_word = tokenizer.sequences_to_texts([prediction])[0]
        # Add next word to the headline. This headline will be used in the next pass of the loop.
        seed_text += " " + next_word
    # Return headline as title-case
    return seed_text.title()
```

## 答案

单击“...”查看答案。

```python
def generate_headline(seed_text, next_words=1):
    for _ in range(next_words):
        # Predict next token
        prediction = predict_next_token(seed_text)
        # Convert token to word
        next_word = tokenizer.sequences_to_texts([prediction])[0]
        # Add next word to the headline. This headline will be used in the next pass of the loop.
        seed_text += " " + next_word
    # Return headline as title-case
    return seed_text.title()
```


```python
seed_texts = [
    'washington dc is',
    'today in new york',
    'the school district has',
    'crime has become']

for seed in seed_texts:
    print(generate_headline(seed, next_words=5))
```

    Washington Dc Is Exempted In Offshore Drilling Plan
    Today In New York Country A New Year For
    The School District Has The World Is The Federal
    Crime Has Become A Informant’S Fate Hangs In


模型经过30次训练后，预测结果可能会有些不足。我们可以注意到，大多数标题具有某种语法意义，但不一定表示出对上下文有良好的理解。通过执行更多次的训练，预测结果可能会有所改善。您可以通过再次运行`fit`单元格，来对模型再训练30次，您应该能看到损失值下降。然后再次测试，结果可能相差很大！

其他改进将是尝试将预训练的嵌入向量与Word2Vec或GloVe一起使用，而不是像在使用Keras嵌入层那样的训练过程中学到它们。有关如何执行此操作的一些信息，请参见[这里](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)。

不过，最重要的是，目前NLP已从简单的LSTM模型转变为基于Transformer的预训练模型。该模型能够从大量文本数据（如Wikipedia）中学习语言的上下文，然后将这些经过预训练的模型用作迁移学习的起点，以解决NLP的任务，例如我们刚刚尝试完成的文本预测任务。您可以尝试使用根据[GPT-2模型](https://openai.com/blog/better-language-models/)实现的[最先进的文本预测器](https://transformer.huggingface.co/doc/gpt2-large)，

要了解有关基于Transformer的模型的更多信息，请阅读有关基于Transformer的双向编码表征（BERT）的[这个博客](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)，并在此DLI课程的“下一步”页面中查找有关其他课程的信息。

## 总结

您完成得不错！我们已成功训练模型预测出了标题中的词，并且使用该模型创建了各种长度的标题。您可以随时试验并生成更多标题。

### 清理显存
继续进行后续内容之前，请执行以下单元，以清理 GPU 显存。


```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```




    {'status': 'ok', 'restart': True}



## 下一步

恭喜，您已经完成本课程的所有学习目标！

作为最后一项练习，同时也为确保您获得本课程证书，请努力解决 [此评估](07_assessment.ipynb) 中的端到端图像分类问题。
