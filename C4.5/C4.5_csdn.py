
import operator
import random
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def ent(data):
    feat = {}
    for feature in data:
        curlabel = feature[-1]
        if curlabel not in feat:
            feat[curlabel] = 0
        feat[curlabel] += 1
    s = 0.0
    num = len(data)
    for it in feat:
        p = feat[it] * 1.0 / num
        s -= p * log(p, 2)
    return s


def remove_feature(data, i, value, flag):
    newdata = []
    for row in data:
        if flag == True:
            if row[i] < value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
        else:
            if row[i] >= value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
#    print('newdata = ',newdata)
    return newdata

# =============================================================================
# 如果是离散值，则使用以下函数进行feature选择
# =============================================================================
# =============================================================================
# def choosebestfeature(data):
#     num = len(data[0]) - 1
#     S = ent(data)
#     maxgain = -1.0
#     bestfeature = -1
#     for i in range(num):
#         curlabel = [it[i] for it in data]
#         curlabel = set(curlabel)
#         if len(curlabel) == 1:
#             continue
#         s = 0.0
#         split = 0.0
#         for value in curlabel:
#             subdata = remove_feature(data,i,value)
#             p = len(subdata) * 1.0 / len(data)
#             s += p * ent(subdata)
#             split -= p * log(p,2)
#         if split == 0:
#             continue
#         gain = (S - s) / split
#         if gain > maxgain:
#             maxgain = gain
#             bestfeature = i
#     return bestfeature
# =============================================================================


def choosebest(data):
    m = len(data)
    maxgain = 0.0
    bestfeature = -1
    bestpoint = -1.0
    n = len(data[0]) - 1
    S = ent(data)
    for i in range(n):
        curfeature = []
        for j in range(m):
            curfeature.append(data[j][i])
        curfeature.sort()
        maxgain = 0.0
        point_id = -1
        for j in range(m - 1):
            point = float(curfeature[j + 1] + curfeature[j]) / 2
            Set = [[it for it in curfeature if it < point],
                   [it for it in curfeature if it > point]]
            p1 = float(len(Set[0])) / m
            p2 = float(len(Set[1])) / m
            split = 0
            if p1 != 0:
                split -= p1 * log(p1, 2)
            if p2 != 0:
                split -= p2 * log(p2, 2)
            if split == 0:
                continue
            gain = (S - p1 * ent(remove_feature(data, i, point, True)) -
                    p2 * ent(remove_feature(data, i, point, False))) / split
            if gain > maxgain:
                maxgain = gain
                bestfeature = i
                bestpoint = point
    return bestfeature, bestpoint


def classify(tree, feature, value):
    if type(tree).__name__ != 'dict':
        return tree
    root = list(tree.keys())[0]
    sons = tree[root]
    i = feature.index(root)
    if value[i] >= list(sons.keys())[1]:
        return classify(sons[list(sons.keys())[1]], feature, value)
    else:
        return classify(sons[list(sons.keys())[0]], feature, value)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def build(data, feature):

    # for it in data:
    #     print(it[-1])

    curlabel = [it[-1] for it in data]
    # print(len(curlabel))
    # print(curlabel)
    '''
    [2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 3.0, 1.0]
    []
    '''
    # curlabel = []
    # for it in data:
    #     curlabel.append(it[-1])
    # print(curlabel)
    '''
    [2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 3.0, 1.0]
    []
    '''
    if curlabel.count(curlabel[0]) == len(curlabel):
        return curlabel[0]
    if len(data[0]) == 1:
        return majorityCnt(curlabel)
    i, point = choosebest(data)
    bestfeature = feature[i]
    tree = {bestfeature: {}}
    del feature[i]
    newfeature = feature[:]
    newdata = remove_feature(data, i, point, True)
    tree[bestfeature][0] = build(newdata, newfeature)
    newdata = remove_feature(data, i, point, False)
    newfeature = feature[:]
    tree[bestfeature][point] = build(newdata, newfeature)
    return tree


def dfs(tree, deep, sample):
    if (type(tree) != sample):
        return deep
    cnt = 0
    for key in tree.keys():
        cnt = max(cnt, dfs(tree[key], deep + 1, sample))
    return cnt


if __name__ == '__main__':
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

    # 训练决策树
    # tree = build(X_train.values.tolist(), X_train.columns.tolist())
    # print(tree)

    # iris = load_iris()
    # train_data = iris['data'][:105]  # X_train.values
    # print(type(train_data))
    # feature = iris['feature_names']  # X_train.columns
    # print(type(feature))

    # print(feature)
    # label = iris['target'][:105]  # y_train.values
    train_data = X_train.values
    # print(type(train_data))
    feature = X_train.columns.tolist()
    # print(type(feature))
    label = y_train.values
    data = train_data.tolist()
    lab = label.tolist()
    test_feature = feature[:]
    num = len(data)

    for i in range(num):
        data[i].append(lab[i])
    tree = build(data, feature)

    # test_data = iris['data'][106:]
    # rest = iris['target'][106:]
    test_data = X_test.values
    rest = y_test.values
    test = test_data.tolist()
    ans = rest.tolist()
    num = len(test_data)
    res = []
    for i in range(num):
        res.append(classify(tree, test_feature, test[i]))
    cnt = 0
    for i in range(num):
        if ans[i] == res[i]:
            cnt += 1
    print('precise = ', cnt * 1.0 / num)
