import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def exc() :
    # method1 : subplot2grid
    plt.figure()
    ax1 = plt.subplot2grid((3 ,3) ,(0 ,0) ,colspan=3) # stands for axes
    ax1.plot([1,2] ,[1,2])
    ax1.set_title('ax1_title')
    ax2 = plt.subplot2grid((3,3) ,(1,0) ,colspan=2)
    ax3 = plt.subplot2grid((3,3) ,(1,2) ,rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.scatter([1,2] ,[2,2])
    ax4.set_xlabel('ax4_x')
    ax4.set_ylabel('ax4_y')
    ax5 = plt.subplot2grid((3,3),(2,1))

    # method 2
    plt.figure()
    gs = gridspec.GridSpec(3,3)
    # use index from 0
    ax6 = plt.subplot(gs[0,:])
    ax7 = plt.subplot(gs[1,:2])
    ax8 = plt.subplot(gs[1: ,2])
    ax9 = plt.subplot(gs[-1,0])
    ax10 = plt.subplot(gs[-1 ,-2])

    # method3 easy to define structure
    f,((ax11 ,ax12), (ax13 ,ax14)) = plt.subplots(2 ,2 ,sharex = True ,sharey = True)
    ax11.scatter([1,2] ,[1,2])
    plt.tight_layout()

    plt.show()
    return None