
"""
自定义奖励函数（包含一个简单分配器）
输入：从data中获取各受灾社区人口、各避难点的容量、距离等；从state获取哪些避难场所被选择；
输出：奖励reward
过程：根据输入信息进行简单分配（就近分配）得到分配结果z，它表示各个受灾社区有多少人分配到各个避难所
    再根据z和data计算reward
    reward=w1r1+w2r2+w3r3(加权求和）
    r1:开放成本; r2:总距离； r3:总覆盖人口
"""
import numpy as np
import pandas as pd

"""
通过权重控制不同方面奖励的重要程度，人员安全最重要，距离次之，成本最后考虑
"""
W1=0.31
W2=0.32
W3=0.37

def get_reward(state, data):

    shelter_number = len(data['shelter']['id'])  # 候选避难场所的总数量
    disaster_number = len(data['disaster']['id'])  # 受灾地区的总数量
    z = np.zeros((disaster_number, shelter_number), np.float)  # 记录分配情况，每行对应一个社区，每列对应一个避难所
    count = len(state)  # 用于记算本次选择的避难所数目
    data_shelter=data['shelter']
    distance = data['connect']['distance']
    distance = distance.to_numpy()
    dmatrix = distance.reshape(disaster_number,shelter_number)
    disavg = np.mean(distance)
    DISTANCE = disavg # 规定disavg千米内是可覆盖的范围
    for i in range(len(state)):  # 删除没有被选中的避难所的信息
        if state[i] == 0:
            count -= 1
            data_shelter = data_shelter.drop(i)
            # data_connect= data_connect.drop(index=data['connect'][(data['connect']['shelterid'] == i)].index)
            # 删除掉没有选中的避难所的距离信息
    if count == 0:
        return 0
    """
    1.分配器：
    对于每个受灾社区i，先按照i到各个避难所的距离从小到大排序，优先选择距离短的避难所，
    但要在满足避难所容量的前提下,所以先要比较剩余容量和社区的人口数
    （这里为了方便，将社区看成一个整体，一个社区的全部人口去到同一个避难所）
    """
    n = disaster_number
    m = 5 #只考虑距离最近的前m个避难所
    data['connect'] = data['connect'].sort_values(by=['disasterid', 'distance'])  # 对每一个受灾社区，将避难场所按距离进行排序

    R=-np.inf

    """
    判断人数是否满足要求，要求选出来的避难所的人数必须能够满足受灾害点的人数避难要求
    其实这里应该考虑一个极端情况，就是选出来的避难所是否能够满足所有灾害点都发生灾害时的避难人数请求
    但是数据集的数据有些问题，所以这里先考虑选出来的避难所能不能满足300个灾害点的避难要求
    """
    sheltersum=data_shelter['避难人数（万人）'].sum()
    # print(sheltersum)
    disastersum=data['disaster']['总户数']
    disastersum=disastersum.sample(n=100)
    disastersum=disastersum.sum()
    # print(disastersum)
    if sheltersum <= disastersum:
        return R


    for i in range(n):  # 对于每一个社区，尽可能安置到距离最近的避难
        ind = np.argsort(dmatrix[i])  # argsort对数组进行排序，并返回排序的索引
        # 对于每个社区i，都得到距离从小到达的排序索引，按照这个顺序进行避难所选择
        assign = False  # 标志该社区是否选定了避难所
        k = 0
        hi = data['disaster'][(data['disaster'].id == i + 1)]['总户数'].item()  # 受灾社区i的人口数（注意i是下标）
        while (assign == False) and (k < m):
            k += 1
            j = ind[k]  # ind[k]的值就是避难所的索引
            if state[j] == 0:
                continue
            if dmatrix[i][j] > DISTANCE:
                continue
            residual_c = data['shelter'][(data['shelter'].id == j + 1)]['避难人数（万人）'].item()  # 避难所j的总容量
            for t in range(n):
                residual_c -= z[t][j]  # 遍历z[][j-1]计算避难所j的剩余容量：总的减去已分配的(注意下标和序号j差1）
            if residual_c > hi or residual_c == hi:
                assign = True
                z[i][j] = hi


    """
        r1:开放成本
        r1=-ΣCostj*IsOpenj
        Costj=避难点j的开放成本
        IsOpenj=避难点j是否开放
    """
    r1=-data_shelter['开放成本（元）'].sum()
    """
        r2:总距离的奖励
        r2=-ΣΣ zij*shortest_dij
        zij=灾害点i被分配到避难所j避难的人数
        shortest_dij=避难点j到灾害点i的距离(目前只考虑了受灾社区和避难场所之间的最短路径距离shortest_distance，
        后续会进一步考虑路径是否畅通以及相关的运输成本等问题)
    """
    r2=0
    for i in range(n):  # 对于每一个社区，
        for j in range(m):  # 对于每一个避难所
            tempdata=data['connect'][(data['connect'].disasterid==i+1) ]
            dij=tempdata[tempdata.shelterid == j+1]['distance'].item() #社区到避难所的距离
            r2+=dij*z[i][j]
    r2=-r2

    """
    r3:覆盖人口的奖励
    r3=Σhi*Is_Coveredi
    hi=灾害点i的人数
    Is_Coveredi=灾害点i是否被覆盖
    """
    Is_Covered = np.zeros(n, np.int)
    # 计算Is_Covered:如果社区i的DISTANCE范围内有避难所开放，他就是被覆盖的
    for i in range(n):  # 对于每一个社区，
        for j in range(m):  # 对于每一个避难所
            tempdata=data['connect'][(data['connect'].disasterid==i+1) ]
            dij=tempdata[tempdata.shelterid == j+1]['distance'].item() #社区到避难所的距离
            if dij < DISTANCE:
                Is_Covered[i] = 1
                continue
    r3 = 0
    for i in range(n):
        hi = data['disaster'][(data['disaster'].id == i + 1)]['总户数'].item()  # 受灾社区i的人口数（注意i是下标）
        r3 += hi * Is_Covered[i]
    r=W1*r1+W2*r2+W3*r3
    return r
