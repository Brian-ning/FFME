import numpy as np

# 信息量计算
q = lambda x: -x*np.log2(x) if np.abs(x)>0.00001 else 0

##############################################
# Deal with normalized mutual information about nonoverlaping community

def nmi_non_olp(coms1, coms2):
    """
    基于标准方法计算两个非重叠社团的归一化互信息量，用于对比两个社团集合的相似程度
    :param coms1: 社团1集合
    :param coms2: 社团2集合
    :return: 归一化互信息值
    """
    # 各社团元素集合
    coms1_nodes = set()
    coms2_nodes = set()

    for com in coms1:
        coms1_nodes.update(com)

    for com in coms2:
        coms2_nodes.update(com)

    # 获取各个社团集合的总规模
    coms1_nodes_num = len(coms1_nodes)
    coms2_nodes_num = len(coms2_nodes)



    # 获取两个社团的总规模
    # 主要用做 P(Ck ∩ Cj) 的分母
    cross_nodes_num = len(coms1_nodes & coms2_nodes)

    # 计算 mutual information

    _H_C1 = 0.0
    _H_C2 = 0.0
    _H_C1_C2 = 0.0

    # calculate H(community1)
    for com in coms1:
        _p = len(com) / coms1_nodes_num
        _H_C1 += q(_p)

    # calculate H(community2)
    for com in coms2:
        _p = len(com) / coms2_nodes_num
        _H_C2 += q(_p)

    # calculate H(community1, community2)
    for com1 in coms1:
        for com2 in coms2:
            _n_com_1_2 = len(set(com1) & set(com2))
            if _n_com_1_2 is not 0:
                _p = _n_com_1_2 / cross_nodes_num
                _H_C1_C2 += q(_p)

    _MI_2 = _H_C1 + _H_C2 - _H_C1_C2

    if abs(_H_C1 + _H_C2) < 0.0001:
        return 0.0
    else:
        return 2 * _MI_2 / (_H_C1 + _H_C2)

##############################################
# Deal with normalized mutual information about overlaping community

def h_x_by_y_norm(coms1, coms2, num):

    h_norm = 0
    for com1 in coms1:
        c1_c = set(com1)
        h_x = q(len(c1_c)/num) + q(1-len(c1_c)/num)
        if abs(h_x) < 0.00001:
            continue
        h_x_by_y_min = 1000000
        for com2 in coms2:
            c2_c = set(com2)
            crs_c = c1_c & c2_c
            p11 = len(crs_c)/num
            p10 = len(c1_c - crs_c)/num
            p01 = len(c2_c - crs_c)/num
            p00 = 1 - p11 - p01 - p10


            h_y = q(p11 + p01) + q(p00 + p10)

            h_x_y = q(p11) + q(p10) + q(p01) + q(p00)


            h_x_by_y = h_x_y - h_y if q(p11)+q(p00)>q(p01)+q(p10) else h_x
            h_x_by_y_min = min(h_x_by_y, h_x_by_y_min)
        h_norm += h_x_by_y_min/h_x

    return h_norm/len(coms1)

def mni_olp_1(coms1, coms2, num = 0):
    """
    计算两个重叠社团集合的互信息量，理论依据来自于文献
    Detecting the overlapping han hierarchical community structure
     in complex networks
    :param coms1: 社团结构1
    :param coms2: 社团结构2
    :param num: 元素个数，赋值可以减少计算量
    :return: 归一化互信息量
    """
    if num == 0:
        num = len(set((x for com in coms1 for x in com)))

    h_x_by_y = h_x_by_y_norm(coms1, coms2, num)
    h_y_by_x = h_x_by_y_norm(coms2, coms1, num)

    return 1 - (h_x_by_y + h_y_by_x)/2



# 包导入控制
__all__ = ['nmi_non_olp','mni_olp_1']