import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from matplotlib import pyplot as plt


train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)

# 查看数据是否有空值
# print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['Age', 'BusinessTravel', 'Department', 'Education', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)

X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition', axis=1),
                                                      train['Attrition'], test_size=0.2,
                                                      random_state=42)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# GBDT
model = GradientBoostingClassifier(random_state=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
y_prob_pred = model.predict_proba(X_valid)[:, 1]
print(accuracy_score(y_valid, y_pred))
print(roc_auc_score(y_valid, y_prob_pred))

# 输出特征重要性
plt.figure()
importance = pd.Series(model.feature_importances_, X_train.columns).sort_values(ascending=False)
importance.plot(kind='bar', title='feature importance')
plt.show()

# predict
predict = model.predict(test)
test['Attrition']=predict
test[['Attrition']].to_csv('submit_gbdt.csv')
