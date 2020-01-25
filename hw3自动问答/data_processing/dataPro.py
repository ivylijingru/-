import pkuseg
import string

# data processing

# train
train_label = []
train_num = []
def load_data():
    '''
    data restored in:
    train_label_normal, train_label_regression
    train_data_original
    return total # of items
    '''
    stwlist = [line.strip() for line in open('stopwords-master/中文停用词表.txt', 'r').readlines()]
    with open( "Data/data/train-set.data", "r") as f0:
        line = f0.readline()
        while line:
            text = line[:-1]

            split = []
            split = text.split('\t', 2)
            #print(split)
            #print("0",split[0])
            #print("1",split[1])
            #print("2",split[2])
            ques = split[0]
            ans = split[1]
            label = split[2]
            #print("ques", ques)
            #print("ans", ans)

            seg = pkuseg.pkuseg()

            ques_tokens = seg.cut(ques)
            #print(ques_tokens)
            ques_without_stopwords = [w for w in ques_tokens if not w in stwlist]
            #print(ques_without_stopwords)

            num = 0
            for i in range(len(ques_without_stopwords)):
            	num += ans.count(ques_without_stopwords[i])

            with open( "demo.txt", 'a') as f1:
                f1.write( str(num) )
                f1.write('\t')
                f1.write(label)
                f1.write( '\n' )

            line = f0.readline()
            
    return train_cnt


if __name__ == "__main__":
    load_data()
