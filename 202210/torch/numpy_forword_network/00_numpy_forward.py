# -*- coding: utf-8 -*-
import numpy as np

# N 是批量大小 D_in是输入维度
# 49/5000 H是隐藏的维度  D_out是输出维度
N ,D_in ,H ,D_out = 64, 1000 ,100 ,10

# 创建随机输入和输出数据
x = np.random.randn(N ,D_in)
y = np.random.randn(N ,D_out)

# 随机初始化权重
w1 = np.random.randn(D_in ,H)
w2 = np.random.randn(H ,D_out)

# 参数的shape
# x[64 ,1000] y[64 ,10] w1[1000 ,100] w2[100 ,10]

learning_rate = 1e-6
for t in range(500):
	# 前向传递，计算预测值y
	h = x.dot(w1)
	h_relu = np.maximum(h ,0)
	y_pred = h_relu.dot(w2)

	# 计算和打印损失loss
	loss = np.square(y_pred - y).sum()
	print(t ,loss)
	
	# 反向传播，计算w1和w2对loss的梯度
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred)
	grad_h_relu = grad_y_pred.dot(w2.T)
	# 参数shape 
	# grad_y_pred[64 ,10] h_relu[64,100] 
	# grad_w2[100 ,10] grad_h_relu[64 ,100]
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0
	grad_w1 = x.T.dot(grad_h)
	# 参数shape 
	# grad_w1 [1000 ,100]
	
	#更新w1和w2的权重
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2
	
	
	