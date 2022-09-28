import matplotlib.pyplot as plt
import numpy as np

def exc():
    # image data
    a = np.array([.3131313131 ,.363636363636 ,.424242424242,
                  .01818181818 ,.626262626262 ,.505050505050,
                  .13131313131 ,.747474747474 ,.1717171717171]).reshape(3 ,3)
    # for the value of 'interpolation' : http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
    # for the value of 'origin' : http://matplotlib.org/examples/pylab_examples/image_origin.html

    plt.imshow(a, interpolation='nearest' ,cmap='bone' ,origin='lower')
    plt.colorbar(shrink=.92)
    plt.xticks()
    plt.yticks()
    plt.show()