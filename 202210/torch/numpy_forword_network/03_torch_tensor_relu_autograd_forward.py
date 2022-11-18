import torch

class MyReLU(torch.autograd.Function):
	"""
	我们通过建立torch.autograd的子类实现自定义的autograd的函数.
	并完成张量的正向和反向计算。
	"""
	
	@staticmethod
	def forward(ctx ,x):
		"""
		在正向传播种，我们接受一个上下文和一个包含输入的张量
		我们必须返回一个包含输出的张量，
		并且我们可以使用上下文对象缓存对象，以便在反向传播种使用
		"""
		ctx.save_for_backward(x)
		return x.clamp(min=0)
		
	@staticmethod
	def backward(ctx ,grad_output):
		"""
		反向传播中，我们接受上下文对象和一个张量
		包含相对正向传播过程的输出的损失和梯度。
		我们可以从上下文对象种检索缓存的数据。
		并且必须计算和返回与正向传播的输入相关的损失函数。
		"""
		x , = ctx.saved_tensors
		grad_x = grad_output.clone()
		grad_x[x < 0] = 0
		return grad_x
		
#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# N是批大小  D_in是输入维度
# H是隐藏层维度 D_out 是输出维度
N ,D_in ,H ,D_out = 64 ,1000 ,100 ,10

# 产生输入和输出的随机张量
x = torch.randn(N ,D_in ,device=device)
y = torch.randn(N ,D_out ,device=device)

# 产生随机权重的张量
w1 = torch.randn(D_in ,H ,device=device ,requires_grad=True)
w2 = torch.randn(H ,D_out ,device=device ,requires_grad=True)

learning_rate = 1e-6
for t in range(650):
	# 正向传播，使用张量上的操作来计算y
	# 我们通过调用MyRelu.apply 函数使用自定义的RelU
	y_pred = MyReLU.apply(x.mm(w1)).mm(w2)
	
	# 计算并输出loss
	loss = (y_pred - y).pow(2).sum()
	print(t ,loss.item())
	
	# 使用autograd计算反向传播过程
	loss.backward()
	
	with torch.no_grad():
		# 用梯度下降更新权重
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad
		
		# 在反向传播之后手动清空梯度
		w1.grad.zero_()
		w2.grad.zero_()
