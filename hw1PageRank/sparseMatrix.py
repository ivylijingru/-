import numpy as np
from scipy import sparse
import random
import csv
from tqdm import tqdm
def compute_PageRank(G, beta=0.85, epsilon=10**-4):
    #Test adjacency matrix is OK
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta #vector
    #Initialize
    ranks = np.ones((n,1))/n #vector
    time = 0
    flag = True
    while flag:
        time +=1
        with np.errstate(divide='ignore'): # Ignore division by 0 on ranks/deg_out_beta
            new_ranks = G.dot((ranks/deg_out_beta)) #vector
        #Leaked PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Stop condition
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False
        ranks = new_ranks
    return(ranks, time)

def takeSecond(elem):
    return elem[1]

if ( __name__ == "__main__"):
    with open('linklist.csv', 'r', newline='', encoding='utf-8-sig') as csvfile:
        lis = []
        lt = []
        node = {}
        crow = []
        ccol = []
        edge = 0
        # rowind = 0
        ii = 0
        spamreader = csv.reader(csvfile)
        for row in tqdm(spamreader):
            if row[0].find(":") > 0:
                continue
            # 清洗掉一部分数据
            lis.append(row)
            lt.append(row[0])
            node[row[0]] = ii
            ii += 1
            if len(lis) >= 1000000:
                break
        ii = 0
        for row in tqdm(lis):
            for item in row[1:]:
                if item in node.keys():
                    crow.append(ii)
                    ccol.append(node[item])
                    edge += 1
            ii += 1
        print("finish building graph!")
        sparse_mat = sparse.csr_matrix((np.bool_(np.ones(edge)),(ccol, crow)), shape=(len(lis), len(lis)))
        pr, iters = compute_PageRank(sparse_mat)
        f = open('a.txt', 'w',encoding='utf-8')
        page_rank = []
        for ii, item in enumerate(lt):
            page_rank.append((item, pr[ii].item(0)))
        page_rank.sort(key=takeSecond)
        for item, pr in page_rank:
            f.write(item+"\t"+str(pr)+"\n")
            # print(item+"\t"+str(pr[ii])+"\n")
        f.close()