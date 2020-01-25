import os
import sys

def trans(input_path, output_path):
    f1 = open(input_path,'r')
    f2 = open(output_path,'w')
    tokens, classes = [],[] 
    for line in f1:
        text, cls = "“" + line[line.find('？')+1:-2].strip() + "”这句话中是否包含对问题“" + line[0:line.find('？')] +"？”的解答？", line[-2]
        f2.write(text + " " + cls + '\n')

data_path = "test-set.data"
output_path = "test-set.txt"
train_g = trans(data_path, output_path)