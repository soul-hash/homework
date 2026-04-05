#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

# 1. 数据加载与预处理 
df = pd.read_csv('winequality-red.csv', sep=';')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 标准化
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# 加偏置项
X = np.c_[np.ones(X.shape[0]), X]

# 划分训练集测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

#  2. 手动线性回归
class LinearRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.W

# 训练
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 3. 手动逻辑回归 
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.W = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.epochs):
            z = X @ self.W
            y_pred = self.sigmoid(z)
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            self.W -= self.lr * dw

    def predict(self, X):
        z = X @ self.W
        y_pred = self.sigmoid(z)
        return np.where(y_pred > 0.5, 1, 0)

# 二分类标签
y_train_cls = np.where(y_train > 6, 1, 0)
y_test_cls = np.where(y_test > 6, 1, 0)

log_reg = LogisticRegression(lr=0.1, epochs=20000)
log_reg.fit(X_train, y_train_cls)
y_pred_log = log_reg.predict(X_test)

# 4. 评估指标 
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

acc = accuracy(y_test_cls, y_pred_log)
prec = precision(y_test_cls, y_pred_log)
rec = recall(y_test_cls, y_pred_log)
f1 = f1_score(y_test_cls, y_pred_log)

# 输出每条测试集的真实分数 & 预测分数
print("="*60)
print("           测试集 红酒质量 真实分数 vs 线性回归预测分数")
print("="*60)
for i in range(len(y_test)):
    print(f"第{i+1:2d}瓶 | 真实质量：{y_test[i]} → 预测评分：{y_pred_lr[i]:.2f}")

# 模型评估结果 
print("\n" + "="*50)
print("线性回归（红酒质量评分预测）评估指标")
print(f"均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")
print("="*50)
print("逻辑回归（红酒好坏分类）评估指标")
print(f"准确率(Accuracy): {acc:.4f}")
print(f"精确率(Precision): {prec:.4f}")
print(f"召回率(Recall): {rec:.4f}")
print(f"F1分数: {f1:.4f}")
print("="*50)


# In[ ]:




