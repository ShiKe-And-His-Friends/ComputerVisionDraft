import numpy as np
import torch

class PointCloud:
    def __init__(self ,points):
        """
        class for manipulating and viewing point clouds
        :param points: <np.float:4 n> Input point cloud matrix
        """
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3 ,:]

    def nbr_points(self):
        """
        Returns the number of points.
        :return: <int> Number of points
        """
        return self.points.shape[1]

    def translate(self ,x):
        """
        Applies a translation to the point cloud
        :param x: <np.float ,3 ,1> translation in x,y,z
        :return: <None>
        """
        for i in range(3):
            self.points[i ,:] = self.points[i ,:] + x[i]

    def rotate(self ,rot_matrix):
        """
        Applies a rotation
        :param rot_matrix: <np.float: 3,3> rotation matrix
        :return: <None>
        """
        self.points[:3 ,:] = np.dot(rot_matrix ,self.points[:3 ,:])
    def transform(self ,transf_matrix):
        """
        Applies a homogeneous transform
        :param transf_matrix: <np.float:4,4> Homogenous transformation matrix
        :return:<None>
        """
        self.points[:3 ,:] = transf_matrix.dot(
            np.vstack((self.points[:3 ,:] ,np.ones(self.nbr_points())))
        )[:3 ,:]

    def normalize(self ,wlh):
        normalizer = [wlh[1] ,wlh[0] ,wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T

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
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name

    def translate(self ,x):
        """
        Applies a tranlations
        :param x: <np.float:3,1> Translation in x,y,z direction
        :return: <None>
        """
        self.center += x

    def rotate(self ,quaternion):
        """
        Rotates box
        :param quaternion: <Quaternion> rorarion to apply
        :return: <None>
        """
        self.center = np.dot(quaternion.rotation_matrix ,self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix ,self.velocity)