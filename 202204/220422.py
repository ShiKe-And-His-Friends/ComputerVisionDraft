import tensorflow as tf
import torch
import os

'''
    Command message: tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Solve : Cpp complier message
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
    CMD nvidia-smi 
'''

x = torch.rand(5 ,3)
print(x)
print(x * x)
print(torch.cuda.is_available())

print("Hello Tensor again")
print(tf.__version__)
print(tf.test.gpu_device_name())

