import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def region_query(X, point_idx, eps):
    # 返回距离point_idx在eps范围内的点索引
    neighbors = []
    for i in range(X.shape[0]):
        if i != point_idx and np.linalg.norm(X[i] - X[point_idx]) <= eps:
            neighbors.append(i)
    return neighbors

def dbscan_dynamic_demand(X, avg_order_qty, eps, min_pts, min_demand, max_demand):
    n = X.shape[0]
    visited = np.zeros(n, dtype=bool)
    cluster_labels = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(X, i, eps)
        # 要求核心点的邻居数量达到min_pts
        if len(neighbors) < min_pts:
            continue
        # 新建簇
        cluster = [i]
        demand_sum = avg_order_qty[i]
        seeds = neighbors.copy()
        for j in seeds:
            if not visited[j] and demand_sum + avg_order_qty[j] <= max_demand:
                visited[j] = True
                new_neighbors = region_query(X, j, eps)
                if len(new_neighbors) >= min_pts:
                    seeds.extend(new_neighbors)
                cluster.append(j)
                demand_sum += avg_order_qty[j]
        # 检查累计需求量是否满足下限
        if demand_sum >= min_demand:
            for idx in cluster:
                cluster_labels[idx] = cluster_id
            cluster_id += 1
        # 不满足下限的点保留为噪声
    return cluster_labels

# 读入数据
df = pd.read_csv(r'F:\distribution classification\cluster_1.csv')
features = df[['longitude', 'latitude']].values
avg_order_qty = df['avg_order_qty'].values

# 参数设置
eps = 0.5            # 距离阈值, 可根据数据分布调整
min_pts = 3          # 最小核心点邻居数
min_demand = 2500    # 最小累计需求量
max_demand = 3000    # 最大累计需求量

labels = dbscan_dynamic_demand(features, avg_order_qty, eps, min_pts, min_demand, max_demand)
df['cluster'] = labels
df.to_csv(r'F:\distribution classification\cluster_1_result.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 读取带聚类标签的数据
df = pd.read_csv(r'F:\distribution classification\cluster_1_result.csv')

# 可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df['longitude'], df['latitude'],
    c=df['cluster'], cmap='tab20', s=30, alpha=0.8
)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN聚类结果可视化')
plt.colorbar(scatter, label='Cluster Label')
plt.show()

# 统计每一簇的avg_order_qty总和
cluster_demand = df.groupby('cluster')['avg_order_qty'].sum().reset_index()
cluster_demand.columns = ['聚类标签', 'avg_order_qty总和']  # 重命名列名

# 按聚类标签排序（-1为噪声点）
cluster_demand = cluster_demand.sort_values(by='聚类标签')

# 输出统计结果
print("每一簇的avg_order_qty总和统计：")
print(cluster_demand)