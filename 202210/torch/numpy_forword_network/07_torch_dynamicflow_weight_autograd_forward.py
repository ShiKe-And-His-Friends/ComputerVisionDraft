import random
import torch

class DynamicNet(torch.nn.Module):
	def __init__(self, D_in ,H ,D_out):
		"""
		构造函数中，我们构建三个nn.Linear函数，它们将在前向传播时被使用
		"""
		super(DynamicNet ,self).__init__()
		self.input_linear = torch.nn.Linear(D_in ,H)
		self.middle_linear = torch.nn.Linear(H ,H)
		self.output_linear = torch.nn.Linear(H ,D_out)
		
	def forward(self ,x):
		"""
		对于模型的前向传播，我们随机选择0 1 2 3
		并重用多次计算隐藏层的middle_linear模块
		由于每个前向传播构建一个动态计算图
		我们可以在定义模型的前向传播时使用常规的Python控制流运算符，如循环和条件语句
		在这里，我们看到，定义计算图时多次重用一个模块是安全的
		这是Lua Torch的一个大改进，因为LuaTorch中每个模块只能使用一次
		"""
		h_relu = self.input_linear(x).clamp(min=0)
		for _ in range(random.randint(0 ,3)):
			h_relu = self.middle_linear(h_relu).clamp(min=0)
		y_pred = self.output_linear(h_relu)
		return y_pred
		
# N是批大小 D是输入维度
# H是隐藏层维度 D_out是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 产生输入和输出随机张量
x = torch.randn(N ,D_in)
y = torch.randn(N ,D_out)

# 实例化上面定义的类来构造我们的类型
model = DynamicNet(D_in ,H ,D_out)

# 构造我们的损失函数(loss function)和优化器(Optimizer)
# 用平凡的随机梯度下降训练这个奇怪的模型是困难的，所以我们使用momentum方法。
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters() ,lr=1e-4 ,momentum=0.9)

for t in range(850):
	# 前向传播： 通过向模型传入x计算预测y
	y_pred = model(x)
	
	# 计算并打印损失
	loss = criterion(y_pred ,y)
	print(t ,loss.item())
	
	# 清零梯度，反向传播，更新权重
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	