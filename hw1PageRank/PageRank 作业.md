## PageRank 作业

**学号：**1700012993	**姓名：**李婧如

### 一、数据预处理

​		观察发现，引用的页面多用 `[[xxx]]` 表示。特殊情况是 `[[xxx|xxx]]` ，此时选择 | 前面的部分。使用正则表达式与 `xml.sax` 库，可提取出引用的内容。此部分存入 csv 文件，每一行是一个页面，第一列为页面标题，其他列为该页面引用的页面标题。但是其中可能出现 `File:/Category:` 等无关信息，用下方的代码去除。同时，由于原文件含有较多的页面，指从中抽取前100万个页面进行实验。

```python
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
```

### 二、 数据结构的选择

最开始实验时选择了 `pygraph.classes.digraph` 库将网页建成有向图，代码如下。然而对该数据结构而言，应用

[教程]: https://www.cnblogs.com/rubinorth/p/5799848.html

中提到的 MapReduce 算法，效率也不如人意。

```python
for row in spamreader:
    if cnt >= 1000000:
        break
    try:
    	dg.add_nodes(row)
    	# cnt += len(row)
    	for ii, item in enumerate(row):
    		if ii != 0:
    			dg.add_edge((row[0], item))
    except:
    	pass
```
于是使用稀疏矩阵的方法进行计算。`scipy` 中的 `spase` 模块有建立稀疏矩阵以及进行稀疏矩阵计算的函数，采用矩阵乘法，多次迭代直到收敛。由运行时间可知应用此方法效率更高。

### 三、PageRank 算法

```python
def compute_PageRank(G, beta=0.85, epsilon=10**-4):
    #检测矩阵大小，G 为稀疏矩阵
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Constants Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta
    ranks = np.ones((n,1))/n
    time = 0
    flag = True
    while flag:
        time +=1
        with np.errstate(divide='ignore'):
            new_ranks = G.dot((ranks/deg_out_beta))
        new_ranks += (1-new_ranks.sum())/n
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False
        ranks = new_ranks
    return(ranks, time)
```

### 四、结果分析

​		得到的 PageRank 结果中，排名较高的是国家与国际组织、大城市、著名人物（如美国总统卢瑟福、奥巴马）、历史中较著名的事件（二战、冷战）等。可能因为与之相关的页面较多（如介绍某人时会介绍其国籍）。排名较低的是一些重定向到其他页面的标题，自身不被引用；或者一些索引式的页面，如 `list of xxx` ，此种页面较多地引用其他页面，自身被引用的频率可能较低。

