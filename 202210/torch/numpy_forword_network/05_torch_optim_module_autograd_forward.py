import torch

# N是批大小 D是输入维度
# H是隐藏层维度 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 产生随机输入和输出张量
x = torch.randn(N ,D_in)
y = torch.randn(N ,D_out)

# 使用nn包定义模型和损失函数
model = torch.nn.Sequential(
	torch.nn.Linear(D_in ,H),
	torch.nn.ReLU(),
	torch.nn.Linear(H ,D_out)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

# 使用optim包定义优化器Optimizer。Optimizer会帮我们更新权重。
# 这里我们使用Adam优化，optim还包含许多别的优化算法。
# Adam构造函数的第一个参数告诉优化器更新哪些张量
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters() ,lr=learning_rate)

for t in range(700):
	y_pred = model(x)
	
	loss = loss_fn(y_pred ,y)
	print(t ,loss.item())
	
	# 反向传播之前，使用optimizer将更新的所有张量的梯度清零（这些梯度是可学习的梯度）
	optimizer.zero_grad()
	
	# 反向传播，根据模型的参数计算loss的梯度
	loss.backward()
	
	# 调用Optimizer的step函数使得所有参数更新
	optimizer.step()