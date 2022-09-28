import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def exc() :
    fig ,ax = plt.subplots()
    x = np.arange(0 ,2*np.pi ,0.01)
    line, = ax.plot(x ,np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i /10.0))
        return line,

    def init():
        line.set_ydata(np.sin(x))
        return line,
    # blit = true does not work on Mac ,set blit = False
    # interval = update frequency

    ani = animation.FuncAnimation(fig = fig ,func=animate ,frames =100,
            init_func=init ,interval=20 ,blit=False)

    # save animation as an mp4 ,requires ffmpeg x264 codes: http://matplotlib.sourceforge.net/api/animation_api.html
    #ani.save('basic_animation.mp4',fps=30 ,extra_args['-vcodec','libx264'])

    plt.show()

    return None