import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 停车点经纬度信息
locations = {
    '东门': (31.2304, 121.4737), '南门': (31.2310, 121.4750), '北门': (31.2320, 121.4760),
    '一食堂': (31.2330, 121.4770), '二食堂': (31.2340, 121.4780), '三食堂': (31.2350, 121.4790),
    '梅苑1栋': (31.2360, 121.4800), '菊苑1栋': (31.2370, 121.4810), '教学2楼': (31.2380, 121.4820),
    '教学4楼': (31.2390, 121.4830), '计算机学院': (31.2400, 121.4840), '工程中心': (31.2410, 121.4850),
    '网球场': (31.2420, 121.4860), '体育馆': (31.2430, 121.4870), '校医院': (31.2440, 121.4880)
}

# 停车点车辆数
vehicle_counts = {
    '东门': 120, '南门': 80, '北门': 100, '一食堂': 90, '二食堂': 110,
    '三食堂': 70, '梅苑1栋': 60, '菊苑1栋': 50, '教学2楼': 40,
    '教学4楼': 30, '计算机学院': 20, '工程中心': 10, '网球场': 5,
    '体育馆': 15, '校医院': 25
}

# 环形图展示停车数量分布
plt.figure(figsize=(8, 8))
plt.pie(vehicle_counts.values(), labels=vehicle_counts.keys(),
        autopct='%.1f%%', startangle=90, wedgeprops={'width': 0.35})
plt.title('校园共享单车停车点车辆分布', fontsize=16)
plt.tight_layout()
plt.show()

# 构建图模型
graph = nx.DiGraph()
points = list(locations.keys())

# 添加边（距离为权重）
for i in range(len(points)):
    for j in range(len(points)):
        if i != j:
            p1, p2 = points[i], points[j]
            dist = geodesic(locations[p1], locations[p2]).km
            graph.add_edge(p1, p2, weight=dist)

# 获取所有最短路径
all_shortest_paths = {}
for start in points:
    for end in points:
        if start != end:
            path = nx.dijkstra_path(graph, start, end, weight='weight')
            all_shortest_paths[(start, end)] = path

# 打印路径
for (start, end), path in all_shortest_paths.items():
    print(f"{start} -> {end}: {path}")

# 可视化调度路径
plt.figure(figsize=(12, 8))
for (start, end), route in all_shortest_paths.items():
    coord_list = np.array([locations[pt] for pt in route])
    plt.plot(coord_list[:, 1], coord_list[:, 0], linestyle='--', marker='o')

# 绘制所有停车点
for name, (lat, lon) in locations.items():
    plt.scatter(lon, lat, color='red', s=60)
    plt.text(lon, lat, name, fontsize=9, ha='right')

plt.title("共享单车最短路径调度图", fontsize=16)
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)
plt.tight_layout()
plt.show()
