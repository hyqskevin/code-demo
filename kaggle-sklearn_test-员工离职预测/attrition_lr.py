import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt


train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

# print(train['Attrition'].value_counts())

"""
No     988
Yes    188
"""

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 查看数据是否有空值
# print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['Age', 'BusinessTravel', 'Department', 'Education', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
lbe_list = []
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)

# print(train)
# train.to_csv('train_label_encoder.csv')

# 训练
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition', axis=1),
                                                      train['Attrition'], test_size=0.2, random_state=42)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
model = LogisticRegression(max_iter=10000,
                           verbose=True,
                           random_state=33,
                           tol=1e-4
                           )

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_valid)[:, 1]
y_pred = list(map(lambda x: 1 if (x >= 0.5) else 0, y_pred))

# 学习曲线
train_size, train_score, valid_score = learning_curve(model, X_train, y_train, scoring='accuracy')

# train size 默认为5次，取平均
mean_train = np.mean(train_score, 1)  # (5,)
# 得到训练得分范围的上下界
upper_train = np.clip(mean_train + np.std(train_score, 1), 0, 1)
lower_train = np.clip(mean_train - np.std(train_score, 1), 0, 1)

mean_test = np.mean(valid_score, 1)
# 得到评估得分范围的上下界
upper_test = np.clip(mean_test + np.std(valid_score, 1), 0, 1)
lower_test = np.clip(mean_test - np.std(valid_score, 1), 0, 1)

# 作图
plt.figure('Fig1')
plt.plot(train_size, mean_train, 'ro-', label='train')
plt.plot(train_size, mean_test, 'go-', label='valid')

# 填充上下界的范围
plt.fill_between(train_size, upper_train, lower_train, alpha=0.2, color='r')
plt.fill_between(train_size, upper_test, lower_test, alpha=0.2, color='g')
# plt.grid()
plt.xlabel('train size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
# plt.savefig('attrition_lr.png')
plt.show()

print(accuracy_score(y_valid, y_pred))

# 预测
predict = model.predict_proba(test)[:, 1]
test['Attrition'] = predict

# 转化为二分类输出
# test['Attrition'] = test['Attrition'].map(lambda x: 1 if x >= 0.5 else 0)
# print(test[['Attrition']])
test[['Attrition']].to_csv('submit_lr.csv')
