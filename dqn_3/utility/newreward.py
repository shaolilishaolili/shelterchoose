"""
自定义规划器 + reward函数
日期：2021年10月12日
规划器：
    输入：已选择的避难所集合
    任务：将n个社区分配到select_count个避难所（一个社区的居民为整体，不做拆分）
    输出：矩阵z(zij就是社区i分配多少人给避难所j，值要么是0，要么是hi)
    思路：就近分配
reward的设计：按照规划器的分配结果z，计算相应的人口覆盖和距离，计算reward
    输出：r=r1+r2
    思路：epsilon约束方法
    r1=Σhi*Is_Coveredi
    r2=if average_dis<=epsilon 0
        else  float('-inf')
    average_dis=Σzij*hij/Σhi
"""

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


DISTANCE = 15  # 规定10千米内是可覆盖的范围  20公里  距离小的话，选出来的避难所应该会多
W1=0.8
W2=0.9
W3=1#通过权重控制不同方面奖励的重要程度，人员安全最重要，距离次之，成本最后考虑

def get_reward(state, data):
    shelter_number = len(data['shelter']['id'])  # 候选避难场所的总数量
    disaster_number = len(data['disaster']['id'])  # 受灾地区的总数量
    z = np.zeros((disaster_number, shelter_number), np.int)  # 记录分配情况，每行对应一个社区，每列对应一个避难所
    count = len(state)  # 用于记算本次选择的避难所数目

    print("state", state)
    data_shelter=data['shelter']
    data_connect = data['connect']
    for i in range(len(state)):  # 删除没有被选中的避难所的信息
        if state[i] == 0:
            count -= 1
            print('drop i',i)
            data_shelter = data_shelter.drop(i)
            print(data['shelter'])
            print(data_shelter)
            data_connect= data_connect.drop(index=data['connect'][(data['connect']['shelterid'] == i)].index)
            # 删除掉没有选中的避难所的距离信息
    if count==0:
        return -np.inf
    # print(data['connect'])
    # print(data_connect)
    """
    1.分配器：
    对于每个受灾社区i，先按照i到各个避难所的距离从小到大排序，优先选择距离短的避难所，
    但要在满足避难所容量的前提下,所以先要比较剩余容量和社区的人口数
    （这里为了方便，将社区看成一个整体，一个社区的全部人口去到同一个避难所）
    """
    n = disaster_number
    m = count  # 为了方便表示
    data['connect'] = data['connect'].sort_values(by=['disasterid', 'shortestd'])  # 对每一个受灾社区，将避难场所按距离进行排序
    for i in range(n):  # 对于每一个受灾社区，尽可能安置到距离最近的避难所
        # 对于每个社区i，按照从上到下的顺序依次选择避难所进行分配（就近分配，最上面的距离最短）
        # 但要在满足避难所容量的前提下,所以先要比较剩余容量和社区的人口数
        temp = data['connect'][(data['connect']['disasterid'] == 1)].reset_index(drop=True)
        assign = False  # 标志该社区是否选定了避难所
        k = 0  # 用于访问社区i到各个避难所的距离
        hi = data['disaster'][(data['disaster'].id == i + 1)]['population'].item()  # 受灾社区i的人口数（注意i是下标）

        while (assign == False) and (k < m):
            j = temp.iloc[k].at['shelterid']  # j的值就是避难所的id
            print("temp",temp)
            print("i",i)
            print("state",state)
            print("j",j)
            residual_c = data['shelter'][(data['shelter'].id == j)]['capacity'].item()  # 避难所j的总容量

            for t in range(n):
                residual_c -= z[t][j - 1]  # 遍历z[][j-1]计算避难所j的剩余容量：总的减去已分配的(注意下标和序号j差1）

            if residual_c> hi or residual_c == hi:
                 assign = True
                 z[i][j - 1] = hi
            k += 1
    print(z)#打印出分配结果

    """
        r1:开放成本
        r1=-ΣCostj*IsOpenj
        Costj=避难点j的开放成本
        IsOpenj=避难点j是否开放
    """
    r1=-data_shelter['opencost'].sum()
    print('r1:',r1)
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
            dij=tempdata[tempdata.shelterid == j+1]['shortestd'].item() #社区到避难所的距离
            r2+=dij*z[i][j]
    r2=-r2
    print('r2:', r2)
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
            dij=tempdata[tempdata.shelterid == j+1]['shortestd'].item() #社区到避难所的距离
            if dij < DISTANCE:
                Is_Covered[i] = 1
                continue
    r3 = 0
    for i in range(n):
        r3 += hi * Is_Covered[i]
    print('r3：',r3)
    r=W1*r1+W2*r2+W3*r3
    print('reward：', r)
    return r











