import torch
import numpy
import os

def pytorch_main():
    print(torch.__version__)
    print(torch.cuda.is_available())

if __name__ == '__main__':
    pytorch_main()
