import networkx as nx
import random


def Multiplex_PartC(m_graph, nx_graphs):
    nodelist = list(m_graph.nodes())
    nodelist.sort()
    node_exit = {}
    layer_num = len(nx_graphs)
    part_coe = {}
    for i in nodelist:
        nodelayer_nei = [0]
        exit_layer = []
        if layer_num > 1:
            for j in range(layer_num-1):
                if i in nx_graphs[j].nodes():
                    exit_layer.append(j)
                    j_nei = set(nx_graphs[j].neighbors(i))
                    for k in range(j+1, layer_num):
                        if i in nx_graphs[k].nodes():
                            k_nei = set(nx_graphs[k].neighbors(i))
                            if j_nei:
                                nodelayer_nei.append(len(j_nei.intersection(k_nei))/len(j_nei))
                            else:
                                nodelayer_nei.append(0)
            if i in nx_graphs[layer_num-1].nodes():
                exit_layer.append(layer_num-1)
            node_exit[i] = list(set(exit_layer))
            part_coe[i] = sum(nodelayer_nei)/((layer_num-1)**2)
        else:
            node_exit[i] = [0]
            part_coe[i] = 0
    return part_coe, node_exit
