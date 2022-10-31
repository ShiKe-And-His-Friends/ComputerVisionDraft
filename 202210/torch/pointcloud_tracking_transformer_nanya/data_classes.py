import numpy as np
import torch

class Box:
    """ Simple data class representing a 3d box including , label ,score and velocity """

    def __init__(self,
            center,
            size ,
            orientation,
            label = np.nan,
            score = np.nan,
            velocity = (np.nan ,np.nan ,np.nan),
            name = None
        ):
        """"
        :param center : [<float>:3] center of box given as x,y,z
        :param size : [<float>:3] size of box in width ,length ,height
        :param orientation : <Quaternion> Box orientation
        :param label: <int> Integer label ,optional
        :param score: <float> Classification socre ,optional
        :param velocity: [<float>:3] box velocity in x , y ,z diection
        :param name:<str> box name ,optinal. Can be used e.g for donate category name
        """

        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        # assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
