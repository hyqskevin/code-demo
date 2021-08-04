## Kaggle 训练: 员工离职预测

问题描述:
员工的各种统计信息，以及该员工是否已经离职，统计的信息包括了（工资、出差、工作环境满意度、工作投入度、是否加班、是否升职、工资提升比例等）
通过训练数据得出员工离职预测，并给出你在测试集上的预测结果。

数据说明:
训练数据和测试数据，保存在train.csv和test.csv文件中
训练集包括1176条记录，31个字段

地址: https://www.kaggle.com/c/bi-attrition-predict/overview

### code demo

tool: numpy, pandas, sklearn, XGBoost, LightGBM

model: logistic regression/SVM/GBDT/XGBoost/LightGBM/*NGBoost/*Catboost

implement: 
- label encoder
- train test split
- model training
- XGBoost, LightGBM param setting
- feature importance plot
- learning curve with upper and lower
- accuracy, roc_auc score