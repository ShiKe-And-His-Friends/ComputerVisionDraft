import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def exc():
    fig = plt.figure()
    ax = Axes3D(fig)
    # X Y value
    X = np.arange(-4 ,4 ,0.25)
    Y = np.arange(-4 ,4 ,0.25)
    X ,Y = np.meshgrid(X ,Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    '''
        X Y Z : data value as 2D arrays
        rstride: array to row stride
        cstride: array to column stride
        color : color of surface patches
        cmap : A colormap for surface patchs
        nrom: an instance of Normalize to map values to colors
        vmin: minimum value to map
        vmax: maximum value to map
        shade: Whether to shade the facecolors
    '''
    ax.plot_surface(X ,Y ,Z ,rstride=1 ,cstride= 1 ,cmap = plt.get_cmap('rainbow'))

    '''
        X Y : data values as numpy.arrays
        Z
        zdir : the direction to use : x y z
        offset: if specified plot a projection of the filled contour on this position in plane normal to zdir
    '''
    ax.contourf(X ,Y ,Z ,zdir='z' ,offset=-2 ,cmap=plt.get_cmap('rainbow'))

    ax.set_zlim(-2 ,2)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())

    plt.show()

    return None