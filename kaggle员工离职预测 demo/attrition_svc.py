import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

train = pd.read_csv('train.csv', index_col=0)
test = test1 = pd.read_csv('test.csv', index_col=0)

# 处理Attrition字段
train['Attrition'] = train['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

# 查看数据是否有空值
# print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr = ['Age', 'BusinessTravel', 'Department', 'Education', 'EducationField',
        'Gender', 'JobRole', 'MaritalStatus',
        'Over18', 'OverTime']

lbe_list = []
for feature in attr:
    lbe = LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)

# train
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition', axis=1), train['Attrition'],
                                                      test_size=0.2, random_state=42)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

# minmax归一化操作
mms = MinMaxScaler(feature_range=(0, 1))
X_train = mms.fit_transform(X_train)
X_valid = mms.fit_transform(X_valid)
test = mms.fit_transform(test)

# 绝对值hinge loss 的SVM
# model = SVC(kernel='rbf',
#             gamma="auto",
#             max_iter=10000,
#             random_state=33,
#             verbose=True,
#             tol=1e-5,
#             cache_size=50000
#             )

# 平方 hinge loss 的SVM
model = LinearSVC(
    max_iter=10000,
    random_state=33,
    verbose=True,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
# y_prob_pred = model.predict_proba(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(acc)

# 学习曲线
train_size, train_score, valid_score = learning_curve(model, X_train, y_train, scoring='accuracy')
mean_train = np.mean(train_score, 1)  # (5,)
mean_valid = np.mean(valid_score, 1)

# 得到训练得分范围的上下界
upper_train = np.clip(mean_train + np.std(train_score, 1), 0, 1)
lower_train = np.clip(mean_train - np.std(train_score, 1), 0, 1)
# 得到评估得分范围的上下界
upper_test = np.clip(mean_valid + np.std(valid_score, 1), 0, 1)
lower_test = np.clip(mean_valid - np.std(valid_score, 1), 0, 1)


# 作图
plt.figure()
plt.plot(train_size, mean_train, 'ro-', label='train')
plt.plot(train_size, mean_valid, 'go-', label='valid')
# 填充上下界的范围
plt.fill_between(train_size, upper_train, lower_train, alpha=0.2, color='r')
plt.fill_between(train_size, upper_test, lower_test, alpha=0.2, color='g')
plt.xlabel('train size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

# predict
predict = model.predict(test)
test1['Attrition'] = predict
test1[['Attrition']].to_csv('submit_svc.csv')
