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
            entropy += probs[y] * np.log2(probs[y])
        return -entropy
    else:
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
            entropy += probs[y] * np.log2(probs[y])
        entropy = -entropy
        # 计算出x中每个类别的个数
        counter = {}
        for x in data_X[feature]:
            counter[x] = counter.get(x, 0) + 1
        # 计算出x中每个类别的概率
        probs = {}
        for x in counter:
            probs[x] = counter[x] / len(data_X[feature])
        # 计算出x的信息熵
        entropy_x = 0
        for x in probs:
            entropy_x += probs[x] * np.log2(probs[x])
        entropy_x = -entropy_x
        # 计算出x和y的联合信息熵
        entropy_xy = 0
        for x in data_X[feature].unique():
            for y in data_y.unique():
                p = len(data_X[(data_X[feature] == x)
                        & (data_y == y)]) / len(data_X)
                if p != 0:
                    entropy_xy += p * np.log2(p)
        entropy_xy = -entropy_xy
        return entropy - (entropy_xy - entropy_x)


def calc_information_gain(data_X, data_y, feature):
    return calc_entropy(data_X, data_y) - calc_entropy(data_X, data_y, feature)


def calc_information_gain_ratio(data_X, data_y, feature):
    return calc_information_gain(data_X, data_y, feature) / calc_entropy(data_X, data_X[feature])


# 基于C4.5算法的决策树构建步骤

# 1.计算数据集D的经验熵H(D)
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
            entropy += probs[y] * np.log2(probs[y])
        return -entropy
    else:
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
            entropy += probs[y] * np.log2(probs[y])
        entropy = -entropy
        # 计算出x中每个类别的个数
        counter = {}
        for x in data_X[feature]:
            counter[x] = counter.get(x, 0) + 1
        # 计算出x中每个类别的概率
        probs = {}
        for x in counter:
            probs[x] = counter[x] / len(data_X[feature])
        # 计算出x的信息熵
        entropy_x = 0
        for x in probs:
            entropy_x += probs[x] * np.log2(probs[x])
        entropy_x = -entropy_x
        # 计算出x和y的联合信息熵
        entropy_xy = 0
        for x in data_X[feature].unique():
            for y in data_y.unique():
                p = len(data_X[(data_X[feature] == x)
                        & (data_y == y)]) / len(data_X)
                if p != 0:
                    entropy_xy += p * np.log2(p)
        entropy_xy = -entropy_xy
        return entropy - (entropy_xy - entropy_x)


# 2.计算特征A对数据集D的经验条件熵H(D|A)
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
            entropy += probs[y] * np.log2(probs[y])
        return -entropy
    else:
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
            entropy += probs[y] * np.log2(probs[y])
        entropy = -entropy
        # 计算出x中每个类别的个数
        counter = {}
        for x in data_X[feature]:
            counter[x] = counter.get(x, 0) + 1
        # 计算出x中每个类别的概率
        probs = {}
        for x in counter:
            probs[x] = counter[x] / len(data_X[feature])
        # 计算出x的信息熵
        entropy_x = 0
        for x in probs:
            entropy_x += probs[x] * np.log2(probs[x])
        entropy_x = -entropy_x
        # 计算出x和y的联合信息熵
        entropy_xy = 0
        for x in data_X[feature].unique():
            for y in data_y.unique():
                p = len(data_X[(data_X[feature] == x)
                        & (data_y == y)]) / len(data_X)
                if p != 0:
                    entropy_xy += p * np.log2(p)
        entropy_xy = -entropy_xy
        return entropy - (entropy_xy - entropy_x)


# 3.计算信息增益g(D,A)=H(D)-H(D|A)
def calc_gain(data_X, data_y, feature):
    return calc_entropy(data_X, data_y) - calc_entropy(data_X, data_y, feature)


# 4.计算信息增益比g_r(D,A)=g(D,A)/H(A)
def calc_gain_ratio(data_X, data_y, feature):
    return calc_gain(data_X, data_y, feature) / calc_entropy(data_X, data_y, feature)


# 5.选择信息增益比最大的特征作为节点特征
def choose_best_feature(data_X, data_y, features):
    best_feature = None
    max_gain_ratio = 0
    for feature in features:
        gain_ratio = calc_gain_ratio(data_X, data_y, feature)
        if gain_ratio > max_gain_ratio:
            max_gain_ratio = gain_ratio
            best_feature = feature
    return best_feature


# 6.如果节点特征的信息增益小于阈值，则置为叶节点，返回
def is_leaf(data_X, data_y, features, threshold):
    if len(features) == 0:
        return True
    if calc_gain(data_X, data_y, features[0]) < threshold:
        return True
    return False


# 7.对节点特征的每一个可能值，根据该特征划分数据集，得到子集Di
def split_data(data_X, data_y, feature):
    data_sets = []
    for value in data_X[feature].unique():
        data_sets.append(data_X[data_X[feature] == value])
    return data_sets


# 8.对每一个子集，以子集为数据集，递归调用步骤1-7，得到子树Ti
def create_tree(data_X, data_y, features, threshold):
    if is_leaf(data_X, data_y, features, threshold):
        return data_y.value_counts().index[0]
    best_feature = choose_best_feature(data_X, data_y, features)
    tree = {best_feature: {}}
    features.remove(best_feature)
    for value in data_X[best_feature].unique():
        sub_data_X = data_X[data_X[best_feature] == value]
        sub_data_y = data_y[data_X[best_feature] == value]
        tree[best_feature][value] = create_tree(
            sub_data_X, sub_data_y, features, threshold)
    return tree


# 9.将子树Ti作为节点的子节点
def fit(data_X, data_y, features, threshold):
    return create_tree(data_X, data_y, features, threshold)


# 11.对新的数据，利用决策树进行分类
def predict(tree, data_X):
    for key in tree.keys():
        value = data_X[key]
        tree = tree[key][value]
        if type(tree) is dict:
            return predict(tree, data_X)
        else:
            return tree


# 12.从根节点开始，对实例进行测试，根据测试结果，选择子树，直到达到叶节点
def score(tree, data_X, data_y):
    y_pred = []
    for i in range(len(data_X)):
        y_pred.append(predict(tree, data_X.iloc[i]))
    y_pred = np.array(y_pred)
    y_true = np.array(data_y)
    return y_pred, y_true


# 13.将实例分到叶节点的类中
def confusion_matrix(y_pred, y_true):
    labels = np.unique(y_true)
    matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(y_pred)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix


# 14.返回类别作为结果
def accuracy_score(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)


# 15.结束
# 16.混淆矩阵
# 17.准确率
# 18.召回率
# 19.F1值
