import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.neighbors import BallTree  # 优化邻域搜索效率

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ----------------------
# 1. 定义带需求量约束的动态密度聚类类（基于OPTICS）
# ----------------------
class DemandConstrainedOPTICS:
    def __init__(self, min_samples=5, demand_min=2500, demand_max=3000):
        self.min_samples = min_samples  # 最小核心点邻域样本数
        self.demand_min = demand_min  # 最小需求量
        self.demand_max = demand_max  # 最大需求量
        self.labels_ = None  # 聚类标签
        self.optics = None  # OPTICS模型实例
        self.tree = None  # 用于高效邻域搜索的BallTree

    def fit(self, X, demands):
        """
        X: 空间特征 (n_samples, 2)，如[x, y]坐标
        demands: 每个样本的需求量 (n_samples,)
        """
        # 构建BallTree加速邻域搜索（适合空间数据）
        self.tree = BallTree(X, metric='euclidean')

        # 1. 用OPTICS计算核心距离和可达距离，获取动态密度信息
        self.optics = OPTICS(
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_method='xi'  # 不自动聚类，仅计算距离
        )
        self.optics.fit(X)
        reachability = self.optics.reachability_
        core_distances = self.optics.core_distances_
        order = self.optics.ordering_  # OPTICS的处理顺序（从高密度到低密度）

        # 2. 初始化标签和访问标记
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1表示未聚类
        visited = np.zeros(n_samples, dtype=bool)
        current_label = 0

        # 3. 按OPTICS顺序处理样本，动态生成聚类
        for idx in order:
            if visited[idx] or core_distances[idx] == -1:
                continue  # 跳过已访问样本或非核心点

            # 动态ε：当前核心点的核心距离（适应局部密度）
            eps = core_distances[idx]

            # 查找ε邻域内的所有样本（用BallTree加速）
            neighbors = self._find_neighbors(X, idx, eps)

            # 从邻域中筛选样本，形成需求量符合约束的簇
            cluster, remaining = self._form_constrained_cluster(
                neighbors, demands, visited
            )

            if len(cluster) > 0:
                self.labels_[cluster] = current_label
                visited[cluster] = True
                current_label += 1

            # 剩余样本标记为未访问，等待后续处理
            visited[remaining] = False

        return self

    def _find_neighbors(self, X, idx, eps):
        """用BallTree高效查找ε邻域内的样本索引"""
        # 返回距离<=eps的样本索引（排除自身）
        indices = self.tree.query_radius(X[idx:idx + 1], r=eps)[0]
        return indices[indices != idx]  # 去掉自身

    def _form_constrained_cluster(self, candidates, demands, visited):
        """从候选样本中构建需求量在[2500, 3000]的簇"""
        # 过滤已访问样本
        candidates = [c for c in candidates if not visited[c]]
        if not candidates:
            return [], []

        # 策略1：优先加入小需求样本，便于凑齐下限
        sorted_small = sorted(candidates, key=lambda x: demands[x])
        current_demand = 0
        cluster = []
        remaining = []

        for c in sorted_small:
            if current_demand + demands[c] <= self.demand_max:
                cluster.append(c)
                current_demand += demands[c]
            else:
                remaining.append(c)

        # 若总需求低于下限，从剩余样本中补充大需求（不超过上限）
        if current_demand < self.demand_min and remaining:
            sorted_large = sorted(remaining, key=lambda x: -demands[x])
            for c in sorted_large:
                if current_demand + demands[c] <= self.demand_max:
                    cluster.append(c)
                    current_demand += demands[c]
                    remaining.remove(c)
                if current_demand >= self.demand_min:
                    break

        return cluster, remaining


# ----------------------
# 2. 读取并预处理数据（关键修改：需求列名为avg_order_qty）
# ----------------------
def load_data(file_path):
    """读取cluster_1.csv，返回空间特征和需求量"""
    df = pd.read_csv(file_path)

    # 空间特征列名（假设为x、y，可根据实际修改）
    spatial_cols = ['longitude', 'latitude']  # 若实际是经纬度，可改为['longitude', 'latitude']
    # 需求量列名改为avg_order_qty（核心修改）
    demand_col = 'avg_order_qty'

    # 检查必要列是否存在
    if not set(spatial_cols + [demand_col]).issubset(df.columns):
        missing = set(spatial_cols + [demand_col]) - set(df.columns)
        raise ValueError(f"数据中缺少必要的列：{missing}，请检查列名！")

    X = df[spatial_cols].values  # 空间特征（n_samples, 2）
    demands = df[demand_col].values  # 需求量（使用avg_order_qty列）

    # 过滤需求量为负或0的异常值
    valid_mask = demands > 0
    X = X[valid_mask]
    demands = demands[valid_mask]
    print(f"加载数据完成，有效样本数：{X.shape[0]}")
    return X, demands, df[valid_mask].copy()  # 返回原始数据（带索引）


# ----------------------
# 3. 执行聚类并分析结果
# ----------------------
if __name__ == "__main__":
    # 数据路径（根据PyCharm项目结构调整）
    file_path = "F:\\distribution classification\\cluster_1.csv"  # 确认路径正确

    # 加载数据（自动读取avg_order_qty作为需求量）
    X, demands, df = load_data(file_path)

    # 聚类参数（可根据数据调整min_samples）
    clf = DemandConstrainedOPTICS(
        min_samples=5,  # 每个核心点的最小邻域样本数（可调整）
        demand_min=2500,  # 最小需求量
        demand_max=3000  # 最大需求量
    )
    clf.fit(X, demands)

    # 将聚类结果添加到原始数据
    df['cluster_label'] = clf.labels_

    # ----------------------
    # 4. 输出聚类统计结果
    # ----------------------
    print("\n聚类结果统计：")
    cluster_stats = df.groupby('cluster_label').agg(
        样本数=('cluster_label', 'count'),
        总需求量=('avg_order_qty', 'sum')  # 统计avg_order_qty的总和
    ).sort_index()
    print(cluster_stats)

    # 检查是否有簇超出需求量范围
    invalid_clusters = cluster_stats[
        (cluster_stats['总需求量'] < 2500) | (cluster_stats['总需求量'] > 3000)
        ]
    if not invalid_clusters.empty:
        print("\n警告：以下簇的需求量超出范围：")
        print(invalid_clusters)

    # ----------------------
    # 5. 可视化聚类结果
    # ----------------------
    plt.figure(figsize=(10, 8))
    unique_labels = set(clf.labels_)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # 离群点（未聚类样本），将s从30调小，例如改为20
            plt.scatter(X[clf.labels_ == label, 0], X[clf.labels_ == label, 1],
                        c='gray', s=10, label='离群点')
        else:
            # 聚类样本，将s从50调小，例如改为30
            plt.scatter(X[clf.labels_ == label, 0], X[clf.labels_ == label, 1],
                        c=[color], s=10, label=f'簇 {label}')

    plt.title('带需求量约束的动态密度聚类结果')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

