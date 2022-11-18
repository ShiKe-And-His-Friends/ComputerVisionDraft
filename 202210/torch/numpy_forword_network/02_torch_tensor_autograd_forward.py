# -*- coding:utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")

# N是批量大小 D_in 是输入维度
# H是隐藏层大小 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 , 100 ,10

# 创建随机Tensor以保持输入和输出
# 设置requires_grad = False 表示不需要计算渐变
# 在向后传播期间求这些Tensor
x = torch.randn(N ,D_in ,device=device ,dtype=dtype)
y = torch.randn(N ,D_out,device=device ,dtype=dtype)

# 创建随机权重Tensors
# 设置requires_grad = True 表示不需要计算渐变
w1 = torch.randn(D_in ,H ,device=device ,dtype=dtype ,requires_grad=True)
w2 = torch.randn(H ,D_out ,device=device ,dtype=dtype ,requires_grad=True)

learning_rate = 1e-6
for t in range(650):
	# 前向传播 适用tensors上的操作计算预测值y
		# 由于w1和w2有requires_grad=True，涉及到这些张量操作让Pytorch构建计算图
	# 从允许自动计算梯度，由于我们不再手工实现反向传播，所以不需要保留中间值的引用
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	
	# 使用Tensors上的操作计算和打印丢失
	# loss是一个形状()的张量
	# loss.item()得到张量对应的python值
	loss = (y_pred - y).pow(2).sum()
	print(t ,loss.item())
	
	# 使用autograd计算反向传播 ,这个调用计算loss对所有requires_grad=True的tensro的梯度 
	# 这次调用后，w1.grad 和w2.grad分别是loss对w1和w2的梯度
	loss.backward()
	
	# 使用梯度下降更新权重。对于这一步，只想对w1和w2的值进行更改；不想更新哦呢阶段构建计算图
	# 所以我们用torch.no_grad()上下文管理器防止Pytorch更新计算图
	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
		
		# 反向传播后手动将梯度设置为零
		w1.grad.zero_()
		w2.grad.zero_()
		