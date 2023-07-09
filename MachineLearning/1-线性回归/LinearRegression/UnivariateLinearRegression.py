# -*- encoding: utf-8 -*-
'''
@File    :   UnivariateLinearRegression.py
@Time    :   2023/07/09 13:56:13
@Author  :   Liao Shuang 
@Function :   《机器学习入门到精通》P14 课程代码 单特征回归模型:根据GDP预测幸福指数
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
# 获取数据集
data = pd.read_csv("E:\\GitLab\\study\\MachineLearning\\1-线性回归\\data\\world-happiness-report-2017.csv")
train_data = data.sample(frac = 0.8) # sample()实现从df中随机抽样,frac是抽取行的比例,返回值为带有N行数据的DataFrame对象
test_data = data.drop(train_data.index)
input_param_name = "Economy..GDP.per.Capita." # X的列名
output_param_name = "Happiness.Score" # Y的列名
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values
# 设定训练参数
num_iterations = 500 # 迭代次数
learning_rate = 0.01 # 学习率
polynomial_degree = 0  
sinusoid_degree = 0 
# 模型训练
linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
(theta, cost_history) = linear_regression.train(alpha = learning_rate, num_iterations = num_iterations)
print(f'最开始的损失值:{cost_history[0]}')
print(f'训练后的损失值:{cost_history[-1]}')
# 绘制损失值变化图
plt.plot(range(num_iterations), cost_history) # 绘制折线图
plt.xlabel("iter")
plt.ylabel("loss")
plt.title("loss function")
plt.show()
# 使用模型进行预测
predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1) #  numpy.linspace(start, end, num=num_points)将在start和end之间生成一个统一的序列,共有num_points个元素
y_predictions = linear_regression.predict(x_predictions)
plt.scatter(x_train, y_train, label = 'train data') # 绘制散点图
plt.scatter(x_test, y_test, label = 'test data')
plt.plot(x_predictions, y_predictions, 'r', label = 'prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("Happy of GDP")
plt.show()