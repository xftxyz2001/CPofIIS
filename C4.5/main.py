import collections
import operator
from math import log

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 调试
def pp(d):
    import pprint
    pprint.pprint(d)
    exit(0)


# 将训练集转换为列表形式
def create_dataset(X_train, y_train):
    dataset = []
    for i in range(len(X_train)):
        dataset.append(list(X_train.iloc[i]) + [int(y_train.iloc[i])])

    # 特征值列表
    labels = list(X_train.columns)

    # 特征对应的所有可能的情况
    labels_full = {}
    for i in range(len(labels)):
        label_list = [example[i] for example in dataset]
        unique_label = set(label_list)
        labels_full[labels[i]] = unique_label

    return dataset, labels, labels_full


# 找到次数最多的类别标签
def majority_count(class_list):
    # 用来统计标签的票数
    class_count = collections.defaultdict(int)

    # 遍历所有的标签类别
    for vote in class_list:
        class_count[vote] += 1

    # 从大到小排序
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的标签
    return sorted_class_count[0][0]


# 计算给定数据集的信息熵(香农熵)
def calc_shannon_entropy(dataset):
    # 计算出数据集的总数
    entries_number = len(dataset)

    # 用来统计标签
    label_counts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for feature_vector in dataset:
        # 得到当前的标签
        current_label = feature_vector[-1]

        # 将对应的标签值加一
        label_counts[current_label] += 1

    # 默认的信息熵
    shannon_entropy = 0.0

    for key in label_counts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(label_counts[key]) / entries_number

        # 以2为底求对数
        shannon_entropy -= prob * log(prob, 2)

    return shannon_entropy


# 按照给定的数值，将数据集分为不大于和大于两部分
def split_dataset4series(dataset, axis, value):
    # 用来保存不大于划分值的集合
    elt_dataset = []
    # 用来保存大于划分值的集合
    gt_dataset = []
    # 进行划分，保留该特征值
    for feat in dataset:
        if feat[axis] <= value:
            elt_dataset.append(feat)
        else:
            gt_dataset.append(feat)

    return elt_dataset, gt_dataset


# 按照给定的特征值，将数据集划分
def split_dataset(dataset, axis, value):
    # 创建一个新的列表，防止对原来的列表进行修改
    ret_dataset = []

    # 遍历整个数据集
    for feature_vector in dataset:
        # 如果给定特征值等于想要的特征值
        if feature_vector[axis] == value:
            # 将该特征值前面的内容保存起来
            reduced_feature_vector = feature_vector[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            # 添加到返回列表中
            ret_dataset.append(reduced_feature_vector)

    return ret_dataset


# 计算连续值的信息增益
def calc_infogain4series(dataset, index, base_entropy):
    # 记录最大的信息增益
    max_infogain = 0.0

    # 最好的划分点
    best_mid = -1

    # 得到数据集中所有的当前特征值列表
    feature_list = [example[index] for example in dataset]

    # 得到分类列表
    class_list = [example[-1] for example in dataset]

    dict_list = dict(zip(feature_list, class_list))

    # 将其从小到大排序，按照连续值的大小排列
    sorted_feature_list = sorted(dict_list.items(), key=operator.itemgetter(0))

    # 计算连续值有多少个
    feature_list_number = len(sorted_feature_list)

    # 计算划分点，保留三位小数
    mid_feature_list = [round((sorted_feature_list[i][0] + sorted_feature_list[i+1][0])/2.0, 3)
                        for i in range(feature_list_number - 1)]

    # 计算出各个划分点信息增益
    for mid in mid_feature_list:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        elt_dataset, gt_dataset = split_dataset4series(dataset, index, mid)

        # 计算两部分的特征值熵和权重的乘积之和
        new_entropy = len(elt_dataset)/len(sorted_feature_list)*calc_shannon_entropy(
            elt_dataset) + len(gt_dataset)/len(sorted_feature_list)*calc_shannon_entropy(gt_dataset)

        # 计算出信息增益
        infogain = base_entropy - new_entropy
        # 计算信息增益率
        infogain = (base_entropy - new_entropy) / base_entropy
        if infogain > max_infogain:
            best_mid = mid
            max_infogain = infogain

    return max_infogain, best_mid


# 计算信息增益
def calc_infogain(dataset, feature_list, current_index, base_entropy):
    # 将当前特征唯一化，也就是说当前特征值中共有多少种
    unique_values = set(feature_list)

    # 新的熵，代表当前特征值的熵
    new_entropy = 0.0

    # 遍历现在有的特征的可能性
    for value in unique_values:
        # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
        sub_dataset = split_dataset(
            dataset=dataset, axis=current_index, value=value)
        # 计算出权重
        prob = len(sub_dataset) / float(len(dataset))
        # 计算出当前特征值的熵
        new_entropy += prob * calc_shannon_entropy(sub_dataset)

    # 计算出“信息增益”
    infogain = base_entropy - new_entropy
    # 计算信息增益率
    infogain = (base_entropy - new_entropy) / base_entropy

    return infogain


# 选择最好的数据集划分特征，根据信息增益值来计算，可处理连续值
def choose_best_splitfeature(dataset, labels):
    # 得到数据的特征值总数
    features_number = len(dataset[0]) - 1

    # 计算出基础信息熵
    base_entropy = calc_shannon_entropy(dataset)

    # 基础信息增益为0.0
    best_infogain = 0.0

    # 最好的特征值
    best_feature = -1

    # 标记当前最好的特征值是不是连续值
    flag_series = 0

    # 如果是连续值的话，用来记录连续值的划分点
    best_series_mid = 0.0

    # 对每个特征值进行求信息熵
    for i in range(features_number):

        # 得到数据集中所有的当前特征值列表
        feature_list = [example[i] for example in dataset]

        if isinstance(feature_list[0], str):
            infogain = calc_infogain(dataset, feature_list, i, base_entropy)
        else:
            # print('当前划分属性为：' + str(labels[i]))
            infogain, best_mid = calc_infogain4series(
                dataset, i, base_entropy)

        # 如果当前的信息增益比原来的大
        if infogain > best_infogain:
            # 最好的信息增益
            best_infogain = infogain
            # 新的最好的用来划分的特征值
            best_feature = i

            flag_series = 0
            if not isinstance(dataset[0][best_feature], str):
                flag_series = 1
                best_series_mid = best_mid

    if flag_series:
        return best_feature, best_series_mid
    else:
        return best_feature


# 创建决策树
def create_tree(dataset, labels):
    # 拿到所有数据集的分类标签
    class_list = [example[-1] for example in dataset]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataset[0]) == 1:
        # 返回剩下标签中出现次数较多的那个
        return majority_count(class_list)

    # 选择最好的划分特征，得到该特征的下标
    best_feature = choose_best_splitfeature(dataset=dataset, labels=labels)

    # 得到最好特征的名称
    best_feature_label = ''

    # 记录此刻是连续值还是离散值,1连续，2离散
    flag_series = 0

    # 如果是连续值，记录连续值的划分点
    mid_series = 0.0

    # 如果是元组的话，说明此时是连续值
    if isinstance(best_feature, tuple):
        # 重新修改分叉点信息
        best_feature_label = str(labels[best_feature[0]]) + \
            '小于' + str(best_feature[1]) + '?'
        # 得到当前的划分点
        mid_series = best_feature[1]
        # 得到下标值
        best_feature = best_feature[0]
        # 连续值标志
        flag_series = 1
    else:
        # 得到分叉点信息
        best_feature_label = labels[best_feature]
        # 离散值标志
        flag_series = 0

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    my_tree = {best_feature_label: {}}

    # 得到当前特征标签的所有可能值
    feature_values = [example[best_feature] for example in dataset]

    # 连续值处理
    if flag_series:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        elt_dataset, gt_dataset = split_dataset4series(
            dataset, best_feature, mid_series)
        # 得到剩下的特征标签
        sub_labels = labels[:]
        # 递归处理小于划分点的子树
        sub_tree = create_tree(elt_dataset, sub_labels)
        my_tree[best_feature_label]['小于'] = sub_tree

        # 递归处理大于当前划分点的子树
        sub_tree = create_tree(gt_dataset, sub_labels)
        my_tree[best_feature_label]['大于'] = sub_tree

        return my_tree

    # 离散值处理
    else:
        # 将本次划分的特征值从列表中删除掉
        del (labels[best_feature])
        # 唯一化，去掉重复的特征值
        unique_values = set(feature_values)
        # 遍历所有的特征值
        for value in unique_values:
            # 得到剩下的特征标签
            sub_labels = labels[:]
            # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            sub_tree = create_tree(split_dataset(
                dataset=dataset, axis=best_feature, value=value), sub_labels)
            # 将子树归到分叉处下
            my_tree[best_feature_label][value] = sub_tree
        return my_tree


def mypredict(input_tree, feature_values):
    # 得到树的根节点
    first_str = list(input_tree.keys())[0]

    # 得到根节点的所有子节点
    second_dict = input_tree[first_str]

    # 得到根节点的特征标签
    feature_label = first_str.split('小于')[0]

    # 得到根节点的划分点
    mid_series = float(first_str.split('小于')[1].split('?')[0])

    # 得到根节点的特征值
    feature_value = feature_values[feature_label]

    # 如果是连续值
    if isinstance(feature_value, float) or isinstance(feature_value, int):
        # 如果当前特征值小于等于划分点
        if feature_value <= mid_series:
            # 得到小于划分点的子树
            value_of_feat = second_dict['小于']
        else:
            # 得到大于划分点的子树
            value_of_feat = second_dict['大于']
    else:
        # 得到当前特征值对应的子树
        value_of_feat = second_dict[feature_value]

    # 如果子树是一个字典的话，说明还没有到叶子节点，继续递归调用
    if isinstance(value_of_feat, dict):
        class_label = mypredict(value_of_feat, feature_values)
    else:
        # 如果子树是一个字符串的话，说明已经到了叶子节点，直接返回当前的类别标签
        class_label = value_of_feat
    return class_label


def mypredicts(input_tree, X_test, labels):
    # 得到测试集的行数
    rows = X_test.shape[0]
    # 存储预测结果
    y_pred = []
    # 遍历每一行
    for i in range(rows):
        # 得到当前行的特征值，结合labels，转换为字典格式
        feature_values = dict(zip(labels, X_test.iloc[i, :]))
        # 得到当前行的预测结果
        y_pred.append(mypredict(input_tree, feature_values))
    # 将预测结果转换为ndarray格式
    y_pred = np.array(y_pred)
    return y_pred


def main():
    data = pd.read_excel('DATASETS_CorkStoppers.xls', sheet_name='Data')
    data.head()

    # 数据预处理
    data = data.dropna()  # 删除空值
    data = data.drop_duplicates()  # 删除重复值
    data = data.drop(['#'], axis=1)
    data.head()

    # 划分训练集和测试集
    y = data['C']
    X = data.drop(['C'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # ID3决策树模型
    id3 = DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train, y_train)
    y_pred_id3 = id3.predict(X_test)

    print('ID3混淆矩阵:\n', confusion_matrix(y_test, y_pred_id3))
    print('分类报告:\n', classification_report(y_test, y_pred_id3))

    # 使用CART算法（可调包）生成决策树模型
    cart = DecisionTreeClassifier()
    cart.fit(X_train, y_train)
    y_pred_cart = cart.predict(X_test)

    print('CART混淆矩阵:\n', confusion_matrix(y_test, y_pred_cart))
    print('分类报告:\n', classification_report(y_test, y_pred_cart))

    # 使用C4.5算法生成决策树模型
    dataset, labels, labels_full = create_dataset(X_train, y_train)
    c45 = create_tree(dataset, labels)
    y_pred_c45 = mypredicts(c45, X_test, labels)
    # print(y_pred_c45)
    # pp(y_pred_c45)

    print('C4.5混淆矩阵:\n', confusion_matrix(y_test, y_pred_c45))
    print('分类报告:\n', classification_report(y_test, y_pred_c45))


if __name__ == '__main__':
    main()

