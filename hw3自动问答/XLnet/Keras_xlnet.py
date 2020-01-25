import os
import sys
from collections import namedtuple
import numpy as np
import pandas as pd
from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI
from keras_radam import RAdam
import time

pretrained_path  = "./chinese_xlnet/"
EPOCH = 10
BATCH_SIZE = 16
SEQ_LEN = 256

MODEL_NAME = 'xlnet_cls.h5'
PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])

config_path = os.path.join(pretrained_path, 'xlnet_config.json')
model_path = os.path.join(pretrained_path, 'xlnet_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'spiece.model')
paths = PretrainedPaths(config_path, model_path, vocab_path)
tokenizer = Tokenizer(paths.vocab)

# 加载模型 #
# Load pretrained model
model = load_trained_model_from_checkpoint(
    config_path=paths.config,
    checkpoint_path=paths.model,
    batch_size=BATCH_SIZE,
    memory_len=0,
    target_len=SEQ_LEN,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)

#### 加载预训练权重
# Build classification model
last = model.output
extract = Extract(index=-1, name='Extract')(last)
dense = keras.layers.Dense(units=768, name='Dense')(extract)
norm = keras.layers.BatchNormalization(name='Normal')(dense)
output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(norm)
model = keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()

##针对下游的finetuning
# 定义优化器，loss和metrics
model.compile(
    optimizer=RAdam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)
### 定义callback函数，只保留val_sparse_categorical_accuracy 得分最高的模型
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("./model/best_xlnet2.h5", monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True,
                            mode='max')

# Read data
class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index):
        s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        return [item[s] for item in self.x], self.y[s]

def generate_sequence(dp):
    global tokenizer
    tokens, classes = [],[]
    f = open(dp,'r')
    i = 0
    for line in f:
        i = i + 1
        text, cls = line[:-3],line[-2]
        encoded = tokenizer.encode(text)[:SEQ_LEN - 1]
        encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
        tokens.append(encoded)
        if cls != "1" and cls != "0":
            cls = "0"
        classes.append(int(cls))
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)

### 读取数据，然后将数据
train_path = "train.txt"
test_path = "test.txt"

### 生成训练集和测试集
train_g = generate_sequence(train_path)
test_g = generate_sequence(test_path)
model.fit_generator(
    generator=train_g,
    validation_data=test_g,
    epochs=EPOCH,
    callbacks=[checkpoint],
)

