import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

# ART	:	Total area of the defects (in pixels)
# N	:	Total number of defects
# PRT	:	Total perimeter of the defects (in pixels)
# ARM	:	Average area of the defects (in pixels)=ART/N
# PRM	:	Average perimeter of the defects (in pixels)=PRT/N
# ARTG	:	Total area of big defects  (in pixels)
# NG	:	Number of big defects (bigger than a specified threshold)
# PRTG	:	Total perimeter of big defects (in pixels)
# RAAR	:	Areas ratio of the defects =ARTG/ART
# RAN	:	Ratio of the number of defects=NG/N


class Node:
    def __init__(self, label, feature_name, feature_value, parent):
        self.label = label
        self.feature_name = feature_name
        self.feature_value = feature_value
        self.parent = parent
        self.children = []


class Tree:
    def __init__(self, feature_names, feature_values, data):
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.data = data
        self.root = None

    def build(self):
        self.root = Node(label=None, feature_name=None,
                         feature_value=None, parent=None)
        self._build(self.root, self.data)

    def _build(self, node, data):
        if data.shape[0] == 0:
            return

        # 如果样本属于同一类别，直接将该节点标记为该类别
        if len(set(data.iloc[:, -1])) == 1:
            node.label = data.iloc[0, -1]
            return

        # 如果样本的所有特征取值相同，直接将该节点标记为样本数最多的类别
        if len(set(data.iloc[:, :-1].values.flatten())) == 1:
            node.label = data.iloc[:, -1].value_counts().index[0]
            return

        # 选择最优划分特征
        feature_name = self._choose_feature(data)
        node.feature_name = feature_name
        feature_index = self.feature_names.index(feature_name)

        # 递归生成子节点
        for feature_value in self.feature_values[feature_index]:
            sub_data = data[data[feature_name] == feature_value]
            child = Node(label=None, feature_name=None,
                         feature_value=feature_value, parent=node)
            node.children.append(child)
            self._build(child, sub_data)

    def _choose_feature(self, data):
        max_gain = 0
        best_feature = None
        for feature_name in self.feature_names:
            gain = self._gain(data, feature_name)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature_name
        return

    def _gain(self, data, feature_name):
        # 计算信息增益
        base_entropy = self._entropy(data.iloc[:, -1])
        feature_index = self.feature_names.index(feature_name)
        feature_values = self.feature_values[feature_index]
        new_entropy = 0
        for feature_value in feature_values:
            sub_data = data[data[feature_name] == feature_value]
            new_entropy += sub_data.shape[0] / data.shape[0] * \
                self._entropy(sub_data.iloc[:, -1])
        return base_entropy - new_entropy

    def _entropy(self, data):
        # 计算信息熵
        entropy = 0
        for label in set(data):
            p = len(data[data == label]) / len(data)
            entropy -= p * math.log(p, 2)
        return entropy

    def predict(self, x):
        node = self.root
        while node.label is None:
            feature_name = node.feature_name
            feature_index = self.feature_names.index(feature_name)
            feature_value = x[feature_index]
            for child in node.children:
                if child.feature_value == feature_value:
                    node = child
                    break
        return node.label

    def print(self):
        self._print(self.root, 0)

    def _print(self, node, depth):
        if node.label is not None:
            print('  ' * depth, node.label)
            return
        print('  ' * depth, node.feature_name, '=', node.feature_value)
        for child in node.children:
            self._print(child, depth + 1)


# 数据预处理
data = pd.read_excel('DATASETS_CorkStoppers.xls', sheet_name='Data')
data = data.dropna()  # 删除空值
data = data.drop_duplicates()  # 删除重复值
data = data.drop(['#'], axis=1)

# 划分训练集和测试集
y = data['C']
X = data.drop(['C'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# ART	:	Total area of the defects (in pixels)
# N	:	Total number of defects
# PRT	:	Total perimeter of the defects (in pixels)
# ARM	:	Average area of the defects (in pixels)=ART/N
# PRM	:	Average perimeter of the defects (in pixels)=PRT/N
# ARTG	:	Total area of big defects  (in pixels)
# NG	:	Number of big defects (bigger than a specified threshold)
# PRTG	:	Total perimeter of big defects (in pixels)
# RAAR	:	Areas ratio of the defects =ARTG/ART
# RAN	:	Ratio of the number of defects=NG/N

# 生成训练集
train_data = pd.concat([X_train, y_train], axis=1)
train_data.columns = ['ART', 'N', 'PRT', 'ARM',
                      'PRM', 'ARTG', 'NG', 'PRTG', 'RAAR', 'RAN', 'C']
train_data['C'] = train_data['C'].map({'OK': 0, 'NG': 1})

# 生成测试集
test_data = pd.concat([X_test, y_test], axis=1)
test_data.columns = ['ART', 'N', 'PRT', 'ARM',
                     'PRM', 'ARTG', 'NG', 'PRTG', 'RAAR', 'RAN', 'C']
test_data['C'] = test_data['C'].map({'OK': 0, 'NG': 1})

# 生成决策树
feature_names = ['ART', 'N', 'PRT', 'ARM',
                 'PRM', 'ARTG', 'NG', 'PRTG', 'RAAR', 'RAN']
feature_values = [
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1]
]



tree = Tree(train_data, feature_names, feature_values)
tree.build()

# 预测
y_pred = []
for i in range(test_data.shape[0]):
    y_pred.append(tree.predict(test_data.iloc[i, :-1]))
y_pred = np.array(y_pred)

# y_pred = []
# for i in range(test_data.shape[0]):
#     y_pred.append(dt.predict(test_data.iloc[i, :-1]))
# y_pred = np.array(y_pred)

# 评估
