import matplotlib.pyplot as plt
import numpy as np

def exc():
    # example 1
    plt.figure(figsize=(6,4))
    # plt.subplot(n_rows ,n_cols ,plot_num)
    plt.subplot(2 ,2 ,1)
    plt.plot([0,1],[0,1])

    plt.subplot(222)
    plt.plot([0,1],[0,2])

    plt.subplot(223)
    plt.plot([0,1],[0,3])

    plt.subplot(224)
    plt.plot([0,1] ,[0,4])
    plt.tight_layout()

    # example 2
    plt.figure(figsize = (6,4))

    plt.subplot(2,1,1)
    plt.plot([0,1],[0,1])

    plt.subplot(234)
    plt.plot([0,1],[0,2])

    plt.subplot(235)
    plt.plot([0,1],[0,3])

    plt.subplot(236)
    plt.plot([0,1],[0,4])
    plt.tight_layout()

    plt.show()
    return None