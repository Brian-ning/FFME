import random
from collections import deque

def random_walk_sampler(Gs, node_exit, sample_size, nodeinfluence, metropolized=False, initial_node=None):
    seqnode = [initial_node] # 初始化当前节点
    d = deque(maxlen=2)
    current_node = initial_node
    curlayer = random.choice(node_exit[current_node]) # 选择某节点所在的层
    G = Gs[curlayer]
    d.append(curlayer)
    while True:
        if len(seqnode) < sample_size: # 游走长度设置
                node_before_step = current_node
                current_node = next_node(G, current_node, metropolized)
                seqnode.append(current_node)
                # 如果节点影响力越大，表示相似性越大，那么就越应该在这两层之间随机游走,
                # 但是为了保证能够把多层都能够采样到，因此需要在其他层也要有相应跳转。
                # 在跳转的时候既要保证相似的层应该频繁的跳转，又要保证不相似的层也应该能够采样对应的节点,同时保证不应该直接“拍平”
                # 如果直接“拍平”类似于层间依赖比较大，可以在不同层之间进行跳转。如果层间依赖为0，那么就不能够拍平，层间跳转不能够
                # 直接从某一层跳转到另外一层。
                if 1 - nodeinfluence[node_before_step] > random.uniform(0,1):
                    if len(set(node_exit[node_before_step])-set(d)) > 0:
                        curlayer = random.choice(list(set(node_exit[node_before_step])-set(d)))
                    else:
                        curlayer = random.choice(node_exit[node_before_step])
                    d.append(curlayer)
                    if current_node not in Gs[curlayer].nodes():
                        current_node = initial_node
        else:
            break
    return seqnode


def next_node(G, current_node, metropolized):
    if metropolized:
        if list(G.neighbors(current_node)):
            candidate = random.choice(list(G.neighbors(current_node)))
            current_node = candidate if (random.random() < float(G.degree(current_node))/G.degree(candidate)) else current_node
    return current_node


def ignore_initial_steps(G, metropolized, excluded_initial_steps, current_node):
    for _ in range(0, excluded_initial_steps):
        current_node = next_node(G, current_node, metropolized)
    return current_node

