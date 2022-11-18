# -*- coding:utf-8 -*-

import torch

dtype = torch.float
device = torch.device("cpu")

# N是批量大小 D_in是输入维度
# H是隐藏维度 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 创建随机输入和输出数据
x = torch.randn(N ,D_in ,device=device ,dtype=dtype)
y = torch.randn(N ,D_out ,device=device ,dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in ,H ,device=device ,dtype=dtype)
w2 = torch.randn(H ,D_out ,device=device ,dtype=dtype)

learning_rate = 1e-6
for t in range(550):
	# 前向协议：计算预测y
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)
	
	# 计算和打印损失
	loss = (y_pred - y).pow(2).sum().item()
	print(t ,loss)
	
	# Backprop计算w1和w2相对的梯度
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h<0] = 0
	grad_w1 = x.t().mm(grad_h)
	
	# 适用梯度下降更新权重
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2