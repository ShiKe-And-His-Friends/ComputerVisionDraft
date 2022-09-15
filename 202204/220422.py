import tensorflow as tf
import torch

'''
    cmd nvidia-smi 
'''

x = torch.rand(5 ,3)
print(x)
print(torch.cuda.is_available())

print("Hello Tensor again")
print(tf.__version__)
print(tf.test.gpu_device_name())

