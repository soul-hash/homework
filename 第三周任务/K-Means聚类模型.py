#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

#  1. 自动加载 Iris 数据集（无需本地文件）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, header=None, names=column_names)

# 提取特征和真实标签
X = df.iloc[:, :-1].values
y_true = df.iloc[:, -1].values

# 把文字标签转成数字（方便评估）
species = np.unique(y_true)
y_true_num = np.array([np.where(species == s)[0][0] for s in y_true])

# 2. 手动实现 K-Means 聚类算法 
class KMeans:
    def __init__(self, k=3, max_iter=100, random_state=42):
        self.k = k                  # 聚类数量（鸢尾花3类，固定k=3）
        self.max_iter = max_iter    # 最大迭代次数
        self.random_state = random_state
        self.centroids = None       # 聚类中心
        self.labels = None          # 聚类结果标签

    # 计算欧氏距离
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # 训练模型
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 随机初始化聚类中心
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        # 迭代优化
        for _ in range(self.max_iter):
            # 步骤1：把每个样本分配到最近的簇
            clusters = [[] for _ in range(self.k)]
            for idx, x in enumerate(X):
                distances = [self.euclidean_distance(x, c) for c in self.centroids]
                cluster_i = np.argmin(distances)
                clusters[cluster_i].append(idx)

            # 步骤2：更新聚类中心
            new_centroids = np.zeros((self.k, n_features))
            for i, cluster in enumerate(clusters):
                new_centroids[i] = np.mean(X[cluster], axis=0)

            # 中心不再变化就提前停止
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        # 最终分配标签
        self.labels = self.predict(X)

    # 预测聚类标签
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, c) for c in self.centroids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

#  3. 训练模型 
kmeans = KMeans(k=3)
kmeans.fit(X)
y_pred = kmeans.labels

# 4. 模型评估
# 轮廓系数
def silhouette_score(X, labels):
    n = len(X)
    scores = []
    for i in range(n):
        # 同一簇内平均距离 a
        same = X[labels == labels[i]]
        a = np.mean([kmeans.euclidean_distance(X[i], s) for s in same])

        # 最近其他簇平均距离 b
        b = float('inf')
        for c in range(kmeans.k):
            if c == labels[i]: continue
            other = X[labels == c]
            dist = np.mean([kmeans.euclidean_distance(X[i], o) for o in other])
            b = min(b, dist)

        # 轮廓系数
        scores.append((b - a) / max(a, b))
    return np.mean(scores)

# 聚类纯度（准确率）
def purity_score(y_true, y_pred):
    mat = np.zeros((3,3))
    for t, p in zip(y_true, y_pred):
        mat[t,p] += 1
    return np.sum(np.max(mat, axis=0)) / len(y_true)

silhouette = silhouette_score(X, y_pred)
purity = purity_score(y_true_num, y_pred)

# 5. 输出完整结果 
print("="*60)
print("                K-Means 鸢尾花聚类 完整结果")
print("="*60)

print("\n【聚类中心】")
for i, center in enumerate(kmeans.centroids):
    print(f"簇 {i} 中心：{center}")

print("\n【每个样本的真实类别 & 聚类预测结果】")
for i in range(len(y_true)):
    print(f"样本{i+1:3d} | 真实品种：{y_true[i]:20s} | 预测簇：{y_pred[i]}")

print("\n【簇与真实品种对应关系】")
for c in range(3):
    mask = y_pred == c
    names, counts = np.unique(y_true[mask], return_counts=True)
    print(f"簇 {c}：{dict(zip(names, counts))}")

print("\n【模型评估指标】")
print(f"轮廓系数：{silhouette:.4f}")
print(f"聚类纯度：{purity:.4f}")
print("="*60)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




