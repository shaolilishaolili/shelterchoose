"""
日期：2021年11月07日
自定义奖励函数
输入：当前的state、Data
输出：奖励reward
过程：
    根据state中的各个避难所是否开放，可以计算r1
    根据state中的分配情况allocij和state中的距离信息，可以计算r2
    根据state中的分配情况zij可以计算被安置的总人口，即r3

    reward=w1r1+w2r2+w3r3(加权求和）
    r1:开放成本; r2:总距离； r3:总覆盖人口
    r1=∑ isopen[j] *cost[j]
    r2=∑ allocij * dij
    r3=∑ allocij 因为已经有了分配信息，可以直接计算被分配的人口数量
"""

W1 = 0.05
W2 = 0.15
W3 = 0.8
def get_reward(state,Data,r_min_max,alloc):
    """
    r_min_max是r1r2r3的上下限
    alloc是分配情况
    """
    count = 0
    isopen = state['is_open']
    for i in range(len(isopen)):
        if isopen[i] == 1:
            count += 1
    if count == 0:
        return -1 #如果没有选中任何避难所直接返回-1

    r1 = 0
    for j in range(len(isopen)):
        if isopen[j] == 1:
            r1 += Data['shelter'].loc[j,'开放成本（元）'] #如果索引为j的避难所开放，加上它的开放成本

    r2=0
    for i in range(alloc.shape[0]):
        for j in range(alloc.shape[1]):
            d = Data['disaster']['id'].to_numpy()
            tempdata = Data['connect'][(Data['connect'].disasterid == d[i])]  # 注意随机选取的100个受灾点，已经不能直接用i索引
            dij = tempdata[tempdata.shelterid == j + 1]['distance'].item()  # 社区到避难所的距离
            r2 += alloc[i][j] * dij

    r3=0
    for i in range(alloc.shape[0]):
        for j in range(alloc.shape[1]):
            r3 += alloc[i][j]

    r1 = (r1 - r_min_max[0]) / (r_min_max[1] - r_min_max[0])
    r2 = (r2 - r_min_max[2]) / (r_min_max[3] - r_min_max[2])
    r3 = (r3 - r_min_max[4]) / (r_min_max[5] - r_min_max[4])
    r = - W1 * r1 - W2 * r2 + W3 * r3

    return r

