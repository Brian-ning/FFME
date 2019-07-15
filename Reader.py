#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, pickle, sys
import networkx as nx
import scipy.io as scio

### Assuming the input files are all pickle encoded networkx graph object ###


def single_readG(path):
    if os.path.isfile(path) and path.endswith(".pickle"):
        g_need = pickle.load(open(path, "rb"))
        #g_need = max(nx.connected_component_subgraphs(g), key=len)
        return g_need
    else:
        sys.exit("##cannot find the pickle file from give path: " + path + "##")


def multi_readG(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        nx_graphs = []
        total_edges = 0
        for name in files:
            if name.endswith(".pickle"):
                ## Serialize to save the object.The Unserialization
                g_need = pickle.load(open(name, "rb"))
                #g_need = max(nx.connected_component_subgraphs(g), key=len)
                nx_graphs.append(g_need)
                total_edges += len(g_need.edges())
        return nx_graphs, total_edges
    else:
        sys.exit("##input path is not a directory##")


def multi_readG_with_Merg(path):
    if os.path.isdir(path):  # Judge whether this path is folder
        files = os.listdir(path)  # Get the file name list under this folder
        nx_graphs = []  # inistall the variable
        m_graph = -1
        total_edges = 0  # The total number of edges
        for name in files:
            if name.endswith("pickle"):  # Checking the file name
                if "merged_graph" in name:
                    m_graph = single_readG(path + '/' + name)
                else:
                    g_need = pickle.load(open(path + '/' + name, "rb"))
                    nx_graphs.append(g_need)
                    total_edges += len(g_need.edges())
        return m_graph, nx_graphs, total_edges


def weight(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        weight_dict = {}
        for name in files:
            if name.endswith('_info.txt'):
                for line in open(path+name):
                    (lay_a, lay_b, coef) = line.split(' ')
                    weight_dict[(lay_a, lay_b)] = float(coef)
    return weight_dict

def true_cluster(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        weight_dict = {}
        for name in files:
            if name.endswith('_true.mat'):
                data = scio.loadmat(path+name)

    return data['s_LNG']


def read_airline(path):
    if os.path.isdir(path):
        print("reading from " + path + "......")
        files  = os.listdir(path)
        nx_graphs = []
        airport_dst = {}
        airport_mapping = {}
        for name in files:
            if name.endswith('_networks.pickle'):
                print("found file " + name + "...")
                graphs = pickle.load(open(path + name, 'rb'))
                for key in graphs:
                    nx_graphs.append(graphs[key])

            elif name.endswith('_Features.pickle'):
                print("found file " + name + "...")
                airport_dst = pickle.load(open(path + name, 'rb'))
            elif name.endswith('_List_mapping.pickle'):
                print("found file " + name + "...")
                airport_mapping = pickle.load(open(path + name, 'rb'))

        #print(len(nx_graphs))
        return nx_graphs, airport_mapping, airport_dst

    else:
        sys.exit('Input path is not a directory')
