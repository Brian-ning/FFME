import networkx as nx
import random


def Multiplex_PartC(m_graph, nx_graphs):
    nodelist = list(m_graph.nodes()) # 获得节点列表
    nodelist.sort()

    node_exit = {} # 初始化节点激活所在的层
    layer_num = len(nx_graphs) # 层的个数
    part_coe = {} # 初始化系数字典

    for i in nodelist:
        nodelayer_nei = {}
        exit_layer = [] # 初始节点所在层集合
        if layer_num > 1: # 当层数是2层的时候，下面的循环会出问题
            for j in range(layer_num-1):
                if i in nx_graphs[j].nodes():
                    exit_layer.append(j)
                    j_nei = set(nx_graphs[j].neighbors(i))
                    for k in range(j+1, layer_num):
                        if i in nx_graphs[k].nodes():
                            k_nei = set(nx_graphs[k].neighbors(i))
                            if j_nei:
                                o_e = len(j_nei.intersection(k_nei))
                                nodelayer_nei[(j,k)] = o_e/(len(j_nei) + len(k_nei)- o_e)
                                # nodelayer_nei.append(len(j_nei.intersection(k_nei))/len(j_nei))
                            else:
                                nodelayer_nei[(j,k)] = 0
            if i in nx_graphs[layer_num-1].nodes():
                exit_layer.append(layer_num-1)
            node_exit[i] = list(set(exit_layer))
            part_coe[i] = nodelayer_nei
        else:
            node_exit[i] = [0]
            part_coe[i] = {}
    return part_coe, node_exit
