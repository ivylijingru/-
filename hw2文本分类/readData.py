import os
import random

from nltk.tokenize import word_tokenize

f = open("train.txt", "r+", encoding='utf-8')
fo = open("glove\\glove.840B.300d.txt", encoding='utf-8')

lines = f.readlines()
line = fo.readlines()

text = []
label = []
for item in lines:
    text.append(item.split("<sep>")[0])
    label.append(item.split("<sep>")[1])

token = []
oov = dict()
dic = dict()
for item in text:
    lt = []
    lt = word_tokenize(item)
    token.append(lt)
    for it in lt:
        dic[it] = 0

word = []
vec = []
lt = [str(random.uniform(-1,1)) for i in range(300)]
# print(lt)
# create a random list in python

for item in line:
    wo = item.split(" ")[0]
    if wo in dic.keys():
        ve = item[:-1].split(" ")[1:]
        dic[wo] = ve

for item in dic.keys():
    if dic[item] == 0:
        oov[item] = lt

for item in oov.keys():
    dic.pop(item)

f.close()
fo.close()

f = open("word_dic.txt", "w", encoding='utf-8')
fo = open("oov.txt", "w", encoding='utf-8')
space = " "
words = []
for k, v in dic.items():
    f.write(k+' '+space.join(v)+'\n')
    words.append(k)
f.close()

for k, v in oov.items():
    fo.write(k+' '+space.join(v)+'\n')
fo.close()

leng = len(words)
# code for oov is leng
f = open("sentence.txt", "w", encoding='utf-8')
for ii, item in enumerate(text):
    lt = []
    ind = []
    lt = word_tokenize(item)
    for it in lt:
        if it in words:
            ind.append(str(words.index(it)))
        else:
            ind.append(str(leng))
    f.write(space.join(ind)+"<sep>"+label[ii])

f.close()