import networkx as nx
import numpy as np
import random
from collections import deque

def forest_fire_sampling(graphs, node_exit, max_sampled_nodes, nodeinfluence, geometric_dist_param=0.85, stratnode=1):
    # 确定当前节点的层
    ch_layer_id = random.choice(node_exit[stratnode])
    sampled_path = [stratnode]
    d = deque(maxlen=2)
    d.append(ch_layer_id)
    # 设置已访问标志
    already_visited = {}
    burn_seed_node = stratnode
    while len(sampled_path) < max_sampled_nodes:
        if burn_seed_node == stratnode or burn_seed_node in already_visited.keys():
            cur_g = graphs[ch_layer_id]
        already_visited[burn_seed_node] = 1

        # 设置燃烧个数和燃烧的节点集
        num_edges_to_burn = np.random.geometric(p=geometric_dist_param)
        neighbors_to_burn = list(cur_g.neighbors(burn_seed_node))[:num_edges_to_burn]

        # 根据当前节点确定是否实现跳转
        if nodeinfluence[burn_seed_node] < random.uniform(0,1): # 值越小，表示在多层中分布越不同；
            if len(set(node_exit[burn_seed_node]) - set(d)) >0:
                ch_layer_id = random.choice(list(set(node_exit[burn_seed_node])-set(d)))
            else:
                ch_layer_id = random.choice(node_exit[burn_seed_node])
            d.append(ch_layer_id)
            neighbors_to_burn.append(burn_seed_node)

        burn_queue = []
        flag = 0
        for n in neighbors_to_burn:

            # 判断在要燃烧的节点中是否存在当前燃烧的跟节点
            if burn_seed_node != n:
                if burn_seed_node != sampled_path[-1]:
                    sampled_path.extend([burn_seed_node, n]) # 加入新的采样节点
                else:
                    sampled_path.append(n) # 加入新的采样节点
                burn_queue.append(n)
            else:
                sampled_path.append(n)
                flag = 1
        if flag == 1:
            burn_queue.append(burn_seed_node)

        # 燃烧队列中的节点
        while len(burn_queue) > 0:
            # 指定队列中的燃烧节点
            burn_seed_node = burn_queue[0]
            burn_queue = burn_queue[1:]
            # 判断这个节点是否已经燃烧过
            if burn_seed_node in already_visited:
                continue
            already_visited[burn_seed_node] = 1

            # 计算下面的邻居燃烧节点
            num_edges_to_burn = np.random.geometric(p=geometric_dist_param)
            neighbors_to_burn = list(cur_g.neighbors(burn_seed_node))[:num_edges_to_burn]
            np.random.shuffle(neighbors_to_burn)

            # 迭代地燃烧节点
            for n in neighbors_to_burn:
                if burn_seed_node != n:
                    if burn_seed_node != sampled_path[-1]:
                        sampled_path.extend([burn_seed_node, n])
                    else:
                        sampled_path.append(n)
                if len(burn_queue) > 0:
                    if stratnode == burn_queue[-1]:
                        final_node = burn_queue.pop()
                        burn_queue.append(n)
                        burn_queue.append(final_node)
    return sampled_path[:max_sampled_nodes]
