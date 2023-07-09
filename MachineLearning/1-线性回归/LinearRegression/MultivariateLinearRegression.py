# -*- encoding: utf-8 -*-
'''
@File    :   MultivariateLinearRegression.py
@Time    :   2023/06/29 16:12:20
@Author  :   Liao Shuang 
@Function :   《机器学习入门到精通》P18 课程代码 多特征回归模型:根据GDP和自由度预测幸福指数
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
from linear_regression import LinearRegression
# 获取数据集
data = pd.read_csv('E:\\GitLab\\study\\MachineLearning\\1-线性回归\\data\world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'
x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values
# 设定训练参数
num_iterations = 500  
learning_rate = 0.01  
polynomial_degree = 0  
sinusoid_degree = 0  
# 模型训练
linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
print('最开始的损失值:',cost_history[0])
print('训练后的损失值:',cost_history[-1])
# 绘制损失值变化图
plt.plot(range(num_iterations), cost_history)
plt.xlabel("iter")
plt.ylabel("loss")
plt.title("loss function")
plt.show()