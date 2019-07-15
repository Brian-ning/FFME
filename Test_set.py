#!/usr/bin/python
# -*- coding: utf-8 -*-

import Reader, pickle
import networkx as nx
import Participation_Coefficient_origal as MPC
import ForestFireCross as ForestFire
import numpy as np
from functools import partial
from multiprocessing import Pool as ThreadPool

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import copy
import warnings
assert gensim.models.doc2vec.FAST_VERSION > -1


edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

default_params = { # 创建一个算法需要用到的参数字典
    'num_walks': 10,                # Number of walks from each node
    'walk_length': 30,              # Walk length， karate datasets is 30
    'window_size': 10,              # Context size for word2vec
    'edge_function': "hadamard",    # Default edge function to use
    "prop_pos": 0.5,                # Proportion of edges to remove nad use as positive samples
    "prop_neg": 0.5,                # Number of non-edges to use as negative samples
    "Reflash_test_data": False        #  (as a proportion of existing edges, same as prop_pos)
}

class Mergeing_vec_N2V:
    def __init__(self, path, ps, j, sampling_p, p, q, num_walks, walk_length, r, num_partitions, dimensions, workers, ff):
        self.path = path
        self.ps = ps
        self.j = j
        self.s_p = sampling_p # 采样比例
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.r = r
        self.exnum = num_partitions
        self.dimensions = dimensions
        self.workers = workers
        self.LG = None
        self.MG = None
        self._pos_edge_list = None
        self._neg_edge_list = None
        self.wvecs = None
        self._rnd = np.random.RandomState(seed=None)  # 随机一个种子
        self.ff = ff


    def generate_pos_neg_links(self):
        # Select n edges at random (positive samples)
        test_g = self.LG[-1] # 选择最后一层作为测试层
        share_L_nodes = set([node for g in self.LG[:-1] for node in g.nodes()]) # 选择训练层所存在的节点
        n_edges = test_g.number_of_edges()
        test_edges = test_g.edges()

        # 如果测试的边都在训练层中，作为最终要测试的边
        share_test_edges = [edge for edge in test_edges if (edge[0] in share_L_nodes) and (edge[1] in share_L_nodes)]
        share_edge = len(share_test_edges)
        npos = int(self.s_p * share_edge) #正采样的边数，默认为1
        nneg = npos                       #负采样的边数，默认等于正采样的边数

        # 负样例的边集
        non_edges = [e for e in nx.non_edges(test_g) if (e[0] in share_L_nodes) and (e[1] in share_L_nodes)]
        print("Finding %d of %d non-edges" % (nneg, len(non_edges)))
        # 随机采样负样例的边表
        rnd_inx = self._rnd.choice(len(non_edges), nneg, replace=False) # 基于之前的随机种子在范围len(non_edges)中生成size为nneg的边数，不用替换；最后形成下标
        neg_edge_list = [non_edges[ii] for ii in rnd_inx] #构建负采样的边

        # 判断网络中负采样的边个数是否满足需要
        if len(neg_edge_list) < nneg:
            raise RuntimeWarning("Only %d negative edges found" % (len(neg_edge_list)))
        print("Finding %d positive edges of %d total edges, which share edges with each layer %d" % (npos, n_edges, share_edge))
        print("Finding %d negtive edges of %d total edges, which share edges with each layer %d" % (npos, n_edges, share_edge))

        # 正例边的构造
        edges = share_test_edges
        pos_edge_list = []
        n_count = 0
        # 随机性的对一个排序进行重排序
        rnd_inx = self._rnd.permutation(share_edge)
        for eii in rnd_inx[:npos]:
            edge = edges[eii]
            pos_edge_list.append(edge)
            n_count += 1

        # 判断正例边的个数是否满足需要
        if len(pos_edge_list) < npos:
            print("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels

    def edges_to_features(self, edge_list, edge_function, dimensions):
        n_tot = len(edge_list)
        feature_vec = np.empty((n_tot, dimensions), dtype='f')

        # Iterate over edges
        for ii in range(n_tot):
            v1, v2 = edge_list[ii]

            # Edge-node features
            emb1 = np.asarray(self.wvecs[str(v1)])
            emb2 = np.asarray(self.wvecs[str(v2)])

            # Calculate edge feature
            feature_vec[ii] = edge_function(emb1, emb2)

        return feature_vec

    def get_alias_edges(self, g, src, dest, p=1, q=1):
        probs = []
        for nei in sorted(g.neighbors(dest)):
            if nei == src:
                probs.append(1 / p)
            elif g.has_edge(nei, src):
                probs.append(1)
            else:
                probs.append(1 / q)
        norm_probs = [float(prob) / sum(probs) for prob in probs]
        return self.get_alias_nodes(norm_probs)

    def get_alias_nodes(self, probs):
        l = len(probs)
        a, b = np.zeros(l), np.zeros(l, dtype=np.int)
        small, large = [], []

        for i, prob in enumerate(probs):
            a[i] = l * prob
            if a[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            sma, lar = small.pop(), large.pop()
            b[sma] = lar
            a[lar] += a[sma] - 1.0
            if a[lar] < 1.0:
                small.append(lar)
            else:
                large.append(lar)
        return b, a

    def preprocess_transition_probs(self, g, directed=False, p=1, q=1):
        alias_nodes, alias_edges = {}, {};
        for node in g.nodes():
            probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
            norm_const = sum(probs)
            norm_probs = [float(prob) / norm_const for prob in probs]
            alias_nodes[node] = self.get_alias_nodes(norm_probs)

        if directed:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edges(g, edge[0], edge[1], p, q)
                # print(alias_edges[edge])
        else:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edges(g, edge[0], edge[1], p, q)
                alias_edges[(edge[1], edge[0])] = self.get_alias_edges(g, edge[1], edge[0], p, q)

        return alias_nodes, alias_edges

    def node2vec_walk(self, g, alias_nodes, alias_edges, walk_length = 30, start = 1):
        path = [start]
        walk_length = self.walk_length
        while len(path) < walk_length:
            node = path[-1]
            neis = sorted(g.neighbors(node))
            if len(neis) > 0:
                if len(path) == 1:
                    l = len(alias_nodes[node][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_nodes[node][1][idx]:
                        path.append(neis[idx])
                    else:
                        path.append(neis[alias_nodes[node][0][idx]])
                else:
                    prev = path[-2]
                    l = len(alias_edges[(prev, node)][0])
                    idx = int(np.floor(np.random.rand() * l))
                    if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                        path.append(neis[idx])
                    else:
                        path.append(neis[alias_edges[(prev, node)][0][idx]])
            else:
                break
        return path

    def learn_embeddings(self, walks, dimensions, window_size=10, niter=5):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        # TODO: Python27 only
        # walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks,
                         size=dimensions,
                         window=window_size,
                         min_count=0,
                         sg=1,
                         workers=self.workers,
                         iter=niter)
        self.wvecs = model.wv

    def run(self):
        cparams = default_params.copy() # 基本参数设置
        path = self.path
        no_reflesh_testdata = bool(1 - cparams["Reflash_test_data"]) #是否刷新：测试-训练数据集
        num_partitions = self.exnum #实验次数
        # Step 1: reading and sampling graphs
        cached_fn = "Sampling_graph/"+ os.path.basename(path) + "%s.graph"
        if os.path.exists(cached_fn) and bool(no_reflesh_testdata):
            # 加载已分好的缓存数据
            print("Loading link prediction graphs from %s" % cached_fn)
            with open(cached_fn, 'rb') as f:
                cache_data = pickle.load(f)
                self.LG = cache_data['g_train']
                self.MG = cache_data['g_merg'] # 合并的网络，可以用于合并层的表示学习
                self._pos_edge_list = cache_data['remove_list']
                self._neg_edge_list = cache_data['ne_list']
        else:
            # 重新构建缓存数据:多层图，合并图，正例边，负例边
            self.MG, self.LG, _ = Reader.multi_readG_with_Merg(path)
            self.generate_pos_neg_links()
            cache_data = {'g_train': self.LG, 'g_merg': self.MG, 'remove_list': self._pos_edge_list, 'ne_list': self._neg_edge_list}
            with open(cached_fn, 'wb') as f:
                pickle.dump(cache_data, f)

        # 构造训练数据集和测试数据集
        nx_graphs_sampled = self.LG[:-1]

        # 构建有向的测试集图例，因为在下面的很多对比算法中有向图作为主要图
        m_graph_sampled = nx.DiGraph()
        for g in nx_graphs_sampled:
            for e in g.edges.data():
                if m_graph_sampled.has_edge(e[0], e[1]):
                    m_graph_sampled[e[0]][e[1]]['weight'] = float(m_graph_sampled[e[0]][e[1]]['weight']) + float(e[2]['weight'])
                else:
                    m_graph_sampled.add_edge(e[0], e[1], weight=e[2]['weight'])

        # 实验：多层网络的表示学习方法(FFS)
        FFSN = [] # 林火采样

        # 将网络中的边权转化为1，然后进行网络的重构
        allnodes = m_graph_sampled.nodes()
        nodeinfluence, node_exit = MPC.Multiplex_PartC(m_graph_sampled, nx_graphs_sampled) #重要性的定义
        ExpansionSample_FFS = partial(ForestFire.forest_fire_sampling, nx_graphs_sampled, node_exit, self.walk_length, nodeinfluence, self.ff)
        with ThreadPool(processes=4) as pool:
            for walks in range(self.num_walks):
                FFSN.extend(pool.map(ExpansionSample_FFS, allnodes)) # VS2: Forest Fire 采样

        visited = []
        visited.append(FFSN)
        Algorithm = ["FFnsME"]
        alg_num = len(visited) # 算法的个数，用于下面算法的评估

        # 对生成的序列进行词向量的学习
        partitioner = model_selection.StratifiedKFold(num_partitions, shuffle=True)

        # 提取要测试的所有边和对应的标签
        edges_all, edge_labels_all = self.get_selected_edges()

        # 基于节点向量计算边权的函数--可以作为一种实验函数
        edge_fn = edge_functions[cparams['edge_function']]

        # 实验性能测试部分
        for steps in range(alg_num):
            auc_train = []
            f1_train = []
            recall_train = []
            acc_train = []
            auc_test = []
            f1_test= []
            recall_test = []
            acc_test = []
            NMI = []
            for train_inx, test_inx in partitioner.split(edges_all, edge_labels_all):
                edges_train = [edges_all[jj] for jj in train_inx]
                labels_train = [edge_labels_all[jj] for jj in train_inx]
                edges_test = [edges_all[jj] for jj in test_inx]
                labels_test = [edge_labels_all[jj] for jj in test_inx]
                # Learn embeddings with current parameter values
                self.learn_embeddings(visited[steps], self.dimensions, window_size=10, niter=5)
                # Calculate edge embeddings using binary function
                edge_features_train = self.edges_to_features(edges_train, edge_fn, self.dimensions)
                edge_features_test = self.edges_to_features(edges_test, edge_fn, self.dimensions)

                # Linear classifier
                scaler = StandardScaler()
                lin_clf = LogisticRegression(C=1)
                clf = pipeline.make_pipeline(scaler, lin_clf)

                # Train & validate classifier
                clf.fit(edge_features_train, labels_train)
                metrics.scorer.mutual_info_scorer(clf, edge_features_train, labels_train)
                auc_train.append(metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train))
                f1_train.append(metrics.scorer.f1_scorer(clf, edge_features_train, labels_train))
                recall_train.append(metrics.scorer.recall_scorer(clf, edge_features_train, labels_train))
                acc_train.append(metrics.scorer.precision_scorer(clf, edge_features_train, labels_train))

                # Test classifier
                NMI.append(metrics.scorer.normalized_mutual_info_scorer(clf, edge_features_train, labels_train))
                auc_test.append(metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test))
                f1_test.append(metrics.scorer.f1_scorer(clf, edge_features_test, labels_test))
                recall_test.append(metrics.scorer.recall_scorer(clf, edge_features_test, labels_test))
                acc_test.append(metrics.scorer.accuracy_scorer(clf, edge_features_test, labels_test))
            print("Algorithm %s; AUC train: %.2f|AUC test: %.4f---F1 train: %.2f|F1 test: %.4f---Recall train:%.2f|Recall test:%.4f---precision train:%.2f|precision test:%.4f : NMI=%.4f"% (Algorithm[steps], sum(auc_train)/num_partitions, sum(auc_test)/num_partitions,sum(f1_train)/num_partitions,sum(f1_test)/num_partitions,sum(recall_train)/num_partitions,sum(recall_test)/num_partitions,sum(acc_train)/num_partitions,sum(acc_test)/num_partitions, sum(NMI)/num_partitions))
        print("---------------END Link Prediction----------------")
