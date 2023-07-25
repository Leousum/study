# 机器学习中的一些基础概念
## 1.变量命名规则与常用变量含义
使用大写字母来表示矩阵,例如X;使用小写字母表示向量,例如y

alpha:α,学习率

theta:θ,模型参数

sigma:σ,标准差

m:样本数量

n:特征数量

上标i:第i行的数据

下标j:第j个特征

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230713170540304.png" alt="image-20230713170540304" style="zoom: 67%;" />

## 2.特征缩放

让每个特征取值都在[-1, 1]这个范围附近

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230714145550612.png" alt="image-20230714145550612" style="zoom: 67%;" />

## 3.均值归一化

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230714145635199.png" alt="image-20230714145635199" style="zoom:67%;" />

## 4.梯度下降和正规方程的选择

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230714155937154.png" alt="image-20230714155937154" style="zoom:67%;" />

## 5.过拟合

**underfit**：欠拟合，**high bias**：高偏差，**high variance**：高方差

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230715222620600.png" alt="image-20230715222620600" style="zoom: 33%;" />

**解决方法**：

<img src="C:\Users\廖双\AppData\Roaming\Typora\typora-user-images\image-20230715223244958.png" alt="image-20230715223244958" style="zoom:50%;" />
