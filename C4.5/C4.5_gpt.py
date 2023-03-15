from collections import Counter
from math import log2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 特征索引
        self.threshold = threshold  # 阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的类别值


class C45DecisionTree:
    def __init__(self, min_samples_split=2, min_impurity=1e-7):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        # list indices must be integers or slices, not numpy.float64
        y = y.astype(int)

        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                # if num_left[c] == 1:
                #     print("class %d at i=%d is new in the left" % (c, i))
                # if num_right[c] == 0:
                #     print("class %d at i=%d is no longer in the right" % (c, i))

                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # 中点

        if best_gini < self.min_impurity:
            return None, None

        Xl, yl, Xr, yr = self._split(X, y, best_idx, best_thr)
        if len(yl) < self.min_samples_split or len(yr) < self.min_samples_split:
            return None, None

        return best_idx, best_thr

    def _split(self, X, y, idx, thr):
        left = np.where(X[:, idx] <= thr)
        right = np.where(X[:, idx] > thr)
        return X[left], y[left], X[right], y[right]

    def _grow(self, X, y):
        impurity = 1.0 - sum((np.sum(y == c) / y.size) **
                             2 for c in range(self.n_classes_))
        if impurity < self.min_impurity:
            return Node(value=Counter(y).most_common(1)[0][0])

        idx, thr = self._best_split(X, y)
        if idx is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        Xl, yl, Xr, yr = self._split(X, y, idx, thr)
        left = self._grow(Xl, yl)
        right = self._grow(Xr, yr)
        return Node(idx, thr, left, right)

    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def print_tree(self, node=None, depth=0):
        if not node:
            node = self.root

        if node.value is not None:
            print("  " * depth, node.value)
        else:
            print("  " * depth, f"X_{node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


data = pd.read_excel('DATASETS_CorkStoppers.xls', sheet_name='Data')

# 数据预处理
data = data.dropna()  # 删除空值
data = data.drop_duplicates()  # 删除重复值
data = data.drop(['#'], axis=1)

# 划分训练集和测试集
y = data['C']
X = data.drop(['C'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# 训练模型
clf = C45DecisionTree()
clf.fit(X_train.values, y_train.values)

# 预测
y_pred = clf.predict(X_test.values)
