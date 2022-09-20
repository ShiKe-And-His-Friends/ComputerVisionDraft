import torch
import torchvision
import numpy
import os

'''
    自动微分
'''
def torch_auto_grad():
    print("torch auto grad")
    #  .grad 自动微分  function 历史纪录
    x = torch.ones(2 ,2 ,requires_grad=True)
    print(x)
    y = x + 2
    print(y)
    print(y.grad_fn)
    z = y * y + 3 * x
    out = z.mean()
    print(z ,out)
    # requires_grad 张量标记
    a = torch.randn(2 ,2)
    a = ((a * 3) / (a - 1))
    print("first grad =",a.requires_grad)
    a.requires_grad_(True)
    print("second grad =",a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)
    # backward 梯度 后向传播
    out.backward()
    print(x.grad) # d(out)/dx
    # autograd 本质雅可比矩阵 Eigen Jacobian
    x = torch.randn(3 ,requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print("y gradient is scalar type=" ,y)
    v = torch.tensor([0.1 ,1.0 ,0.001] ,dtype = torch.float)
    y.backward(v)
    print("y gradient is jacobian type=" ,x.grad)
    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)

'''
    环境 && Tensor
'''
def pytorch_enviroment():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch)
    print('//////////////////////////////////////////////////////////////')
    # 数据类型
    m = torch.rand(3 ,5)
    x = torch.empty(3 ,5)
    z = torch.zeros(3 ,5)
    print(m)
    print(x)
    print(z)
    # 类型转换
    x = x.new_ones(3 ,5 ,dtype = torch.double)
    print(x)
    x = torch.randn_like(x , dtype = torch.float32)
    print(x)
    # 算数运算
    result = torch.empty(3 ,5)
    torch.add(x ,m ,out=result)
    print("result=" ,result)
    result.add_(result)
    print("add result=", result)
    print("row all , cloums index=" ,result[:,1])
    # Tensor改类型
    o = torch.rand(5 ,3)
    print("o=" ,o)
    oo = o.view(15)
    ooo = o.view(-1 ,15)
    print(o.size() , oo.size() ,ooo.size())
    # Tensor选元素
    x = torch.rand(1)
    print("x=", x)
    print(x.item())
    x = torch.rand(10)
    print("x=", x)
    print(x.tolist())
