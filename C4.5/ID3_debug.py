import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_id3))
print('Classification Report:\n', classification_report(y_test, y_pred_id3))
