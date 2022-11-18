import torch

class TwoLayerNet(torch.nn.Module):
	def __init__(self ,D_in ,H ,D_out):
		"""
		我们在构造函数实现两个nn.Linear模块，并将它们作为成员变量。
		"""
		super(TwoLayerNet ,self).__init__()
		self.linear1 = torch.nn.Linear(D_in ,H)
		self.linear2 = torch.nn.Linear(H ,D_out)
	
	def forward(self ,x):
		"""
		在前向传播的函数种，我们接受一个输入的张量，也必须返回一个输出张量
		我们可以使用构造函数定义模块以及张量的任意（可微分）的操作
		"""
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred
	
# N是批大小 D_in是输入维度
# H是隐藏层维度 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 产生输入和输出的随机张量
x = torch.randn(N ,D_in)
y = torch.randn(N ,D_out)

# 通过实例化上面定义的类来构建我们的模型
model = TwoLayerNet(D_in ,H ,D_out)

# 构造损失函数和优化器
# SGD构造函数中对model.parameters（）的调用。
# 将包含模型的一部分，即两个nn.Linear模块的可学习参数
loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters() ,lr=1e-4)
for t in range(750):
	# 前向传播：通过模型传递x计算预测y
	y_pred = model(x)
	
	# 计算并输出loss
	loss = loss_fn(y_pred ,y)
	print(t ,loss.item())
	
	# 清零梯度，反向传播，更新权重
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()