#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import xml.sax
import re
from pygraph.classes.digraph import digraph

# ==================================================文本解析部分===============================================

class MovieHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.CurrentData = ""
        self.text = ""
        self.title = ""
        self.fo = open("foo.txt", "w", encoding="utf-8")
        self.dic = {}
        self.pattern = re.compile('\[\[([^:\[\]\|]+)\]\]')
        self.pattern2 = re.compile('\[\[([^:\[\]]+)\|')

   # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag

    # 元素结束事件处理
    def endElement(self, tag):
        self.CurrentData = ""

   # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "text":
            self.text = content
         # self.fo.write(str(self.pattern.findall(content)))
         # self.fo.write(content)
         # findall 返回一个列表。将该有向边插入到图中。
         # 重复加边的话没关系吗
            try:
                dg.add_nodes(self.pattern.findall(content))
                dg.add_nodes(self.pattern2.findall(content))
                for item in self.pattern.findall(content):
                    dg.add_edge((self.title, item))
                for item in self.pattern2.findall(content):
                    dg.add_edge((self.title, item))
            except:
                pass
             # self.dic[self.title].extend(self.pattern.findall(content))
             # self.dic[self.title].extend(self.pattern2.findall(content))
        if self.CurrentData == "title":
            self.title = content
            # self.dic[self.title] = []
            try:
                dg.add_nodes([self.title])
            except:
                pass
            # self.fo.write(self.title+"\n")

# ================================================== Page Rank ===============================================

class PRIterator:
    __doc__ = '''计算一张图中的PR值'''

    def __init__(self, dg):
        self.damping_factor = 0.85  # 阻尼系数,即α
        self.max_iterations = 100  # 最大迭代次数
        self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
        self.graph = dg

    def page_rank(self):
        #  先将图中没有出链的节点改为对所有节点都有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
        damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分

        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):  # 遍历所有“入射”的页面
                    rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank

            print("This is NO.%s iteration" % (i + 1))
            print(page_rank)

            if change < self.min_delta:
                flag = True
                break
        if flag:
            print("finished in %s iterations!" % node)
        else:
            print("finished out of 100 iterations!")
        return page_rank

# ================================================== main 函数 ===============================================


# if __name__ == '__main__':
#     dg = digraph()

#     dg.add_nodes(["A", "B", "C", "D", "E"])

#     dg.add_edge(("A", "B"))
#     dg.add_edge(("A", "C"))
#     dg.add_edge(("A", "D"))
#     dg.add_edge(("B", "D"))
#     dg.add_edge(("C", "E"))
#     dg.add_edge(("D", "E"))
#     dg.add_edge(("B", "E"))
#     dg.add_edge(("E", "A"))

#     pr = PRIterator(dg)
#     page_ranks = pr.page_rank()

#     print("The final page rank is\n", page_ranks)

if ( __name__ == "__main__"):
   
    global dg
    dg = digraph()
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = MovieHandler()
    parser.setContentHandler( Handler )

    parser.parse("enwiki-20190920-pages-articles-multistream.xml")
    print("finish building graph")
    pr = PRIterator(dg)
    page_ranks = pr.page_rank()

    print("The final page rank is\n", page_ranks)