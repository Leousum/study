"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float) # astype()对数据类型进行转换

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0) # 标准差(Standard Deviation,简称SD或者STD)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
