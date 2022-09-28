import matplotlib.pyplot as plt
import numpy as np

def exc():
    x = np.linspace(-3 ,3 ,50)
    y1 = 2 * x + 1
    y2 = x ** 2

    plt.figure()
    plt.xlim((-1,2))
    plt.ylim((-2 ,3))
    # set new ticks
    new_ticks = np.linspace(-1 ,2 ,5)
    plt.xticks(new_ticks)
    #set tick labels
    plt.yticks([-2 ,-1.8 ,-1 ,1.22 ,3],
               [r'$really\ bad$' ,r'$bad$' ,r'$normal$' ,r'$good$' ,r'$really\ good$'])
    l1, = plt.plot(x ,y1 ,label='linear line')
    l2, = plt.plot(x ,y2 ,color='red' ,linewidth=1.0 ,linestyle='--' ,label='square line')
    plt.legend(loc='upper right')
    # plt.legend(handles=[l1 ,l2] ,labels=['up' ,'down'],loc='best')
    # legend(handles=(line1 ,line2 line3),
    #       labels = ('label1' ,'label2' ,'label3'),
    #       lco = 'upper right')
    # loc ACCEPTS:
    #       'best':0
    #       'upper right':1
    #       'upper left':2
    #       'lower left':3
    #       'lower right':4
    #       'right':5
    #       'center left':6
    #       'center right':7
    #       'lower center':8
    #       'upper center':9
    #       'center':10
    #       """

    plt.show()