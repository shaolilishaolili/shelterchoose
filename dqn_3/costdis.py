"""
这是对数据进行处理，首先按照论文中的根据面积计算成本的方法将成本计算出来并写进表中，
然后是用geopy库，利用表中已有的经纬度计算出两个点之间的数据，并写进表中，这个可以构造
出来connect表
"""

import pandas as pd
from geopy.distance import geodesic
import numpy as np
Data = {}
dataset_path = './data/ddata.xlsx'  # 原始数据集
dataset = ['disaster', 'shelter', 'connect']

for file in dataset:
    Data[file] = pd.read_excel(dataset_path, sheet_name=file)

def insertCost():
    area = Data['shelter']['有效使用面积（公顷）']
    opencost = []
    for i in area:
        if i >= 1.5:
            opencost.append(40 * 10000 * i)
        elif i >= 0.2 and i < 1.5:
            opencost.append(15 * 10000 * i)
        elif i >= 0.03 and i < 0.2:
            opencost.append(5 * 10000 * i)
    opencost = pd.DataFrame(opencost)
    Data['shelter']['开放成本（元）'] = opencost



def insertDis():
    shelter_lng = Data['shelter']['lng']
    shelter_lat = Data['shelter']['lat']
    shelter_lng = shelter_lng.to_numpy()
    shelter_lat = shelter_lat.to_numpy()
    shelter_lnglat = [(x, y) for x in shelter_lat for y in shelter_lng]

    disaster_lng = Data['disaster']['lng']
    disaster_lat = Data['disaster']['lat']
    disaster_lng = disaster_lng.to_numpy()
    disaster_lat = disaster_lat.to_numpy()
    disaster_lnglat = [(x, y) for x in disaster_lat for y in disaster_lng]

    dis = []
    x = []
    y = []
    for i in range(int(len(disaster_lnglat) ** 0.5)):
        for j in range(int(len(shelter_lnglat) ** 0.5)):
            x.append(i + 1)
            y.append(j + 1)
            geodesic(disaster_lnglat[i], shelter_lnglat[j]).m
            dis.append(geodesic(disaster_lnglat[i], shelter_lnglat[j]).m)
    print(len(dis))
    print(dis)
    print(max(dis))
    df1 = pd.DataFrame({'key': range(len(dis)),
                        'disasterid': x})
    df2 = pd.DataFrame({'key': range(len(dis)),
                        'shelterid': y})
    df3 = pd.DataFrame({'key': range(len(dis)),
                        'distance': dis})
    df = pd.merge(df1, df2, on='key')
    df = pd.merge(df, df3, on='key')
    print(df1)
    print(df2)
    print(df3)
    print(df)
    Data['connect'] = df


insertCost()
insertDis()
