# build test data
from nltk.tokenize import word_tokenize
import wordninja
from nltk.corpus import stopwords

f = open("test.txt", "r+", encoding='utf-8')
fo = open("final_data.txt", "w", encoding='utf-8')
fi = open("word_dic.txt", "r+", encoding='utf-8')
stop_words = set(stopwords.words('english')) 
lines = f.readlines()
text = []
for item in lines:
    text.append(item[:-1])
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
        spl = wordninja.split(it)
        for iit in spl:
            if iit not in stop_words:
                if iit in word:
                    ind.append(str(word.index(iit)))
                else:
                    ind.append(str(leng))
    fo.write(space.join(ind)+"\n")

f.close()
fo.close()
fi.close()