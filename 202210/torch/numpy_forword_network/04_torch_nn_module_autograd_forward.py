# -*- coding:utf-8 -*-
import torch

# N是批大小 D是输入维度
# H是隐藏层维度 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 创建输入和输出随机张量
x = torch.randn(N ,D_in)
y = torch.randn(N ,D_out)

# 使用nn包将我们模型定义一些列的层
# nn.Sequential是包含其他模块的模块，按顺序产生其输出
# 每个线性模块使用线性函数从输入计算输出，并保存内部的权重和张量
# 在构建模型之后，使用.to()方法将其移动到所需设备 
model = torch.nn.Sequential(
	torch.nn.Linear(D_in ,H),
	torch.nn.ReLU(),
	torch.nn.Linear(H ,D_out),
)

# nn包还包含常用损失函数的定义
# 在这种情况下，我们将使用平均平方误差(MSE)作为损失函数
# 设置reduction="sum" ，表示我们计算的平方误差的“和”，而不是平均值
# 实践中，常使用均方根误差reduction="elemenwise_mean"来使用均方误差来作为损失
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-4
for t in range(650):
	# 前向传播： 通过模型传入x计算预测y
	# 模块对象重载__call__运算符，可以像函数那样调用他们
	# 这样相当于传一个张量，然后返回一个输出张量
	y_pred = model(x)
	
	# 计算并打印损失
	#  传递包含y的预测值和真实值的张量，损失函数会返回损失的张量
	loss = loss_fn(y_pred ,y)
	print(t ,loss.item())
	
	# 反向传播之前清零梯度
	model.zero_grad()
	
	# 反向传播：计算模型的损失对所有学习参数的导数（梯度）
	# 在内部，每个模块的参数存储在requires_grad=True的张量种
	# 因此调用将计算模型种所有可学习参数的梯度
	loss.backward()
	
	# 使用梯度下降更新权值
	# 每个参数都是张量，所以我们可以像以前那样得到它的数值和梯度
	with torch.no_grad():
		for param in model.parameters():
			param -= learning_rate * param.grad