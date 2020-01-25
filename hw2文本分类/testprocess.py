# build test data
from nltk.tokenize import word_tokenize

f = open("self_test.txt", "r+", encoding='utf-8')
fo = open("test_data.txt", "w", encoding='utf-8')
fi = open("word_dic.txt", "r+", encoding='utf-8')

lines = f.readlines()
text = []
label = []
for item in lines:
    text.append(item.split("<sep>")[0])
    label.append(item.split("<sep>")[1])

word = []
line = fi.readlines()
for item in line:
    word.append(item.split(" ")[0])
leng = len(word)
space = " "
for ii, item in enumerate(text):
    lt = []
    ind = []
    lt = word_tokenize(item)
    for it in lt:
        if it in word:
            ind.append(str(word.index(it)))
        else:
            ind.append(str(leng))
    fo.write(space.join(ind)+"<sep>"+label[ii])

f.close()
fo.close()
fi.close()