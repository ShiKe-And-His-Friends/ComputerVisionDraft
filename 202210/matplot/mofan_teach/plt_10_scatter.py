import matplotlib.pyplot as plt
import numpy as np

def exc():
    n = 1024
    X = np.random.normal(0 ,1 ,n)
    Y = np.random.normal(0 ,1 ,n)
    T = np.arctan2(Y ,X) # for color later on

    plt.scatter(X, Y ,s=75 ,c =T ,alpha=.5)
    plt.xlim(-1.5 ,1.5)
    plt.xticks(()) # ignore ticks
    plt.ylim(-1.5 ,1.5)
    plt.yticks(())  # ignore ticks

    plt.show()