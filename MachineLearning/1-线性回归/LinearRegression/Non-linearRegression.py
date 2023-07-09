# -*- encoding: utf-8 -*-
'''
@File    :   Non-linearRegression.py
@Time    :   2023/07/09 17:13:53
@Author  :   Liao Shuang 
@Function :   《机器学习入门到精通》P19 课程代码 非线性回归模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
# 获取数据集
data = pd.read_csv('E:\\GitLab\\study\\MachineLearning\\1-线性回归\\data\\non-linear-regression-x-y.csv')
x = data[['x']].values.reshape((data.shape[0], 1))
y = data[['y']].values.reshape((data.shape[0], 1))
data.head(10)
plt.plot(x, y)
plt.show()
# 设定训练参数
num_iterations = 50000  
learning_rate = 0.02  
polynomial_degree = 15  
sinusoid_degree = 15  
normalize_data = True
# 模型训练 
linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)
print(f'最开始的损失值:{cost_history[0]}')
print(f'训练后的损失值:{cost_history[-1]}')
# 绘制损失值变化图
plt.plot(range(num_iterations), cost_history) # 绘制折线图
plt.xlabel("iter")
plt.ylabel("loss")
plt.title("loss function")
plt.show()
# 使用模型进行预测
predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1);
y_predictions = linear_regression.predict(x_predictions)
plt.scatter(x, y, label='Training Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.show()