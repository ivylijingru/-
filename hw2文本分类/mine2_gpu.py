import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

#textCNN模型
class textCNN(nn.Module):
    def __init__(self,args):
        super(textCNN, self).__init__()
        vocb_size = args['vocb_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix=args['embedding_matrix']
        #需要将事先训练好的词向量载入
        self.embeding = nn.Embedding(vocb_size, dim,_weight=embedding_matrix)
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,64,64)
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2)
                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.embeding(x)
        x=x.view(x.size(0),1,max_len,word_dim)
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        #print(x.size())
        output = self.out(x)
        return output

    def save(self):
        prefix = "check_points/"
        if not os.path.isdir(prefix):
            os.mkdir(prefix)
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        print('model name', name.split('/')[-1] )
        torch.save(self.state_dict(), name)
        torch.save(self.state_dict(), prefix+'latest.pth')

#====================================read from file===================================
nb_words = 0
max_len = 0
n_class = 0
word_dim = 0

word_dic = open("word_dic2.txt", "r+", encoding='utf-8')
oov = open("oov2.txt", "r+", encoding='utf-8')
text_id = open("sentence2.txt", "r+", encoding='utf-8')
test_id = open("test_data2.txt", "r+", encoding='utf-8')

word = word_dic.readlines()
oovList = oov.readlines()
text = text_id.readlines()
test = test_id.readlines()

nb_words = len(word)+1
n_class = 3
word_dim = 300
max_len = 284
embedding_matrix = np.zeros((nb_words, word_dim))

texts_with_id=np.zeros([len(text),max_len])
test_with_id=np.zeros([len(test),max_len])

label_x = []
label_y = []
lis = []
for i, lines in enumerate(oov):
    if i == 0:
        lis = lines[:-1].split(" ")[1:]
    else:
        break

for i, lines in enumerate(word):
    lt = lines[:-1].split(" ")[1:]
    for j in range(len(lt)):
        embedding_matrix[i][j] = float(lt[j])

# set up the vector for words not in dic (a random vector)
for j in range(len(lis)):
    embedding_matrix[len(word)][j] = float(lis[j])

for i, lines in enumerate(text):
    label_x.append(int(lines.split("<sep>")[1]))
    lt = lines.split("<sep>")[0].split(" ")
    for j in range(len(lt)):
        texts_with_id[i][j] = float(lt[j])

for i, lines in enumerate(test):
    label_y.append(int(lines.split("<sep>")[1]))
    lt = lines.split("<sep>")[0].split(" ")
    for j in range(len(lt)):
        test_with_id[i][j] = float(lt[j])

word_dic.close()
oov.close()
text_id.close()
test_id.close()

args={}

#========================================================textCNN调用的参数================================================
args['vocb_size']=nb_words
args['max_len']=max_len
args['n_class']=n_class
args['dim']=word_dim

EPOCH=30;

args['embedding_matrix']=torch.Tensor(embedding_matrix)

#==========================================================构建textCNN模型==================================================
cnn=textCNN(args)
device = torch.device('cuda')
cnn.to(device)

LR = 0.0005
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#损失函数
loss_function = nn.CrossEntropyLoss()
#训练批次大小
epoch_size=100
texts_len=len(texts_with_id)
print(texts_len)
#划分训练数据和测试数据
# x_train, x_test, y_train, y_test = train_test_split(texts_with_id, label, test_size=0.2, random_state=42)
x_train, y_train = texts_with_id, label_x
x_test, y_test = test_with_id, label_y

test_x=torch.LongTensor(x_test)
test_y=torch.LongTensor(y_test)
train_x=x_train
train_y=y_train

test_epoch_size=300;
for epoch in range(EPOCH):

    # for i in tqdm(range(0,(int)(len(train_x)/epoch_size))):
    for i in range(0,(int)(len(train_x)/epoch_size)):
        b_x = Variable(torch.LongTensor(train_x[i*epoch_size:min(i*epoch_size+epoch_size,len(train_x))]))
        b_y = Variable(torch.LongTensor((train_y[i*epoch_size:i*epoch_size+epoch_size])))
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(str(i))
        # print(loss)
        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (b_y == pred_y)
        acc = acc.cpu()
        acc = acc.numpy().sum()
        accuracy = acc / (b_y.size(0))

    acc_all = 0;
    
    pred = np.array([])
    b = np.array([])

    for j in range(0, (int)(len(test_x) / test_epoch_size)):
        b_x = Variable(torch.LongTensor(test_x[j * test_epoch_size:j * test_epoch_size + test_epoch_size]))
        b_y = Variable(torch.LongTensor((test_y[j * test_epoch_size:j * test_epoch_size + test_epoch_size])))
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        test_output = cnn(b_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        # print(pred_y)
        # print(test_y)
        acc = (pred_y == b_y)
        acc = acc.cpu()
        acc = acc.numpy().sum()
        # print("acc " + str(acc / b_y.size(0)))
        acc_all = acc_all + acc
        pred_y = pred_y.cpu()
        b_y = b_y.cpu()

        pred = np.concatenate((pred, pred_y), axis = 0)
        b = np.concatenate((b, b_y), axis = 0)
    
    print(precision_recall_fscore_support(pred, b, average='macro'))
    print(precision_recall_fscore_support(pred, b, average='micro'))

    cnn.save()
    #accuracy = acc_all / (test_y.size(0))
    #print("epoch " + str(epoch) + " step " + str(i) + " " + "acc " + str(accuracy))