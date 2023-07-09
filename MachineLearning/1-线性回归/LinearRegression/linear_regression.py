# -*- encoding: utf-8 -*-
'''
@File    :   linear_regression.py
@Time    :   2023/06/28 16:04:16
@Author  :   Liao Shuang 
@Function :   《机器学习入门到精通》P10 课程代码
'''
import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self, data, lables, polynominal_degree = 0, sinusoid_degree = 0, normalize_data = True) -> None:
        '''
        1.对数据做预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        polynominal_degree:特征变换复杂度(次数)
        sinusoid_degree:sin(x)中x的度数
        normalize_data:是否对数据进行标准化
        '''
        # 预处理完成后的数据、平均值、标准差
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynominal_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.labels = lables
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynominal_degree = polynominal_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        number_features = self.data.shape[1] # 特征数量
        self.theta = np.zeros((number_features, 1)) # 创建0矩阵

    def train(self, alpha, num_iterations = 500):
        '''
        训练函数,执行梯度下降函数
        parm:alpha 学习率
        parm:num_iterations 迭代次数
        '''
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        '''
        梯度下降函数:实际迭代模块,会迭代num_iterations次
        '''
        cost_history = list() # 损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history
    
    def gradient_step(self, alpha):
        '''
        损失计算方法:梯度下降参数更新函数,注意是矩阵运算
        '''
        num_examples = self.data.shape[0] # 样本个数
        prediction = LinearRegression.hypothesis(self.data, self.theta) # 预测值,这里就体现了是一个线性模型
        delta = prediction - self.labels
        self.theta = self.theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T # 根据公式得到的新θ

    @staticmethod
    def hypothesis(data, theta):
        '''
        预测函数
        '''
        prediction = np.dot(data, theta)
        return prediction
    
    def cost_function(self, data, labels):
        '''
        损失值计算函数
        '''
        num_examples = data.shape[0] # 样本个数
        delta = LinearRegression.hypothesis(data, self.theta) - labels # 预测值
        cost = ((1 / 2) * np.dot(delta.T, delta)) / num_examples # 损失值,损失值不应该和样本大小相关,应该是一个平均值
        return cost[0][0]
    
    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data, self.polynominal_degree, self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_processed, labels)
    
    def predict(self, data):
        '''
        用训练好的参数模型,去预测得到回归值结果
        '''
        data_processed = prepare_for_training(data, self.polynominal_degree, self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions