##*********************#

## https://github.com/MorvanZhou/tutorials/tree/master/matplotlibTUT

## Thanks for your share.

##*********************#
import matplotlib.pyplot as plt
import numpy as np

def exc() :
    print("easy example.")
    x = np.linspace(-1 ,1 ,50)
    # y = 2 * x + 1
    y = x ** 2
    plt.plot(x ,y)
    plt.show()

