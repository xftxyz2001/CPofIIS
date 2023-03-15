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


class Node:
    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y
        self.feature = None
        self.sub_nodes = None
        self.clazz = None


# 计算信息熵的函数：计算Infor(D)
def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))


# 定义计算信息增益的函数：计算g(D|A)
def g(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: infor(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    # 计算Infor(D|A)
    e2 = sum(e1 * p1)
    return infor(data[str2]) - e2


# 定义计算信息增益率的函数：计算gr(D,A)
def gr(data, str1, str2):
    return g(data, str1, str2) / infor(data[str1])


def calc_entropy(data_X, data_y, feature=None):
    if feature == None:
        # 计算出y中每个类别的个数
        counter = {}
        for y in data_y:
            counter[y] = counter.get(y, 0) + 1
        # 计算出y中每个类别的概率
        probs = {}
        for y in counter:
            probs[y] = counter[y] / len(data_y)
        # 计算出y的信息熵
        entropy = 0
        for y in probs:
            entropy -= probs[y] * np.log2(probs[y])
        return entropy
    else:
        return 0
        # # 计算出data_X中每个特征值的个数
        # counter = {}
        # for x in data_X[feature]:
        #     counter[x] = counter.get(x, 0) + 1
        # # 计算出data_X中每个特征值的概率
        # probs = {}
        # for x in counter:
        #     probs[x] = counter[x] / len(data_X[feature])
        # # 计算出data_X的信息熵
        # entropy = 0
        # for x in probs:
        #     entropy -= probs[x] * np.log2(probs[x])
        # return entropy


def build_tree(X_train, y_train):
    assert len(X_train) == len(y_train)

    node = Node(X_train, y_train)

    # 1. 如果样本集合为空，返回空
    if len(node.data_X) == 0:
        return None

    # 2.如果样本属于同一类别，返回该类别
    if len(set(node.data_y)) == 1:
        node.clazz = node.data_y[0]
        return node

    # 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分；：把当前结点标记为叶结点，并将其类别设定为该结点所含样本最多的类别；

    # 3. 计算最大信息增益率
    # max_gain = 0
    # max_feature = None
    # for feature in node.data_X.columns:
    #     gain = calc_gain(node.data_X, node.data_y, feature)
        # if gain > max_gain:
        #     max_gain = gain
        #     max_feature = feature
    # gains = np.zeros(len(node.data_X.columns))
    # 初始信息熵
    entropy_start = calc_entropy(node.data_X, node.data_y)
    entropys = {}
    gains = {}
    # print(gains)
    for feature in node.data_X.columns:
        # print(feature)
        # gain = calc_gain(node.data_X, node.data_y, feature)
        entropys[feature] = calc_entropy(node.data_X, node.data_y, feature)
        gains[feature] = entropy_start - entropys[feature]
    # max_gains = max(gains)

    # 找到gains中最大的值对应的key
    max_feature = max(gains, key=gains.get)
    # 如果maxx_feature的信息增益为0，返回空
    if gains[max_feature] == 0:
        # 计算node.data_y中每个元素的个数
        counter = {}
        for y in node.data_y:
            counter[y] = counter.get(y, 0) + 1
        # 返回数量最多的那个类别
        node.clazz = max(counter, key=counter.get)
        return node

    # 4. 如果最大信息增益小于阈值，返回空
    # if max_gain < 0.1:
        # return None

    # 5. 根据最大信息增益的特征，划分子节点
    node.feature = max_feature
    node.sub_nodes = {}
    for value in set(node.data_X[max_feature]):
        sub_data_X = node.data_X[node.data_X[max_feature] == value]
        sub_data_y = node.data_y[node.data_X[max_feature] == value]
        sub_node = Node(sub_data_X, sub_data_y)
        node.sub_nodes[value] = build_tree(sub_node)

    return node


tree = build_tree(X_train, y_train)
