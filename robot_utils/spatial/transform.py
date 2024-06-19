# Author: Sophie Lueth
# Date: December 2023

import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    """
    Helper class for SE3 transforms

    from robot_utils.spatial.transform import Transform
    """
    def __init__(self, translation=np.zeros((3,)), rotation=Rotation.from_quat([0, 0, 0, 1])):
        """
        Args:
            translation (np.ndarray of shape (3,)): Translation of Homogenous Transform
            rotation (Rotation): rotation of Homogenous Transform
        """
        self._translation = translation
        self._rotation = rotation

        self._matrix = np.eye(4)
        self._matrix[:3,:3] = rotation.as_matrix()
        self._matrix[:3, 3] = translation

    def to_matrix(self):
        return self._matrix
    
    @classmethod
    def from_matrix(cls, matrix):
        """
        creates Transform from 4x4 homogenous Transformation Matrix
        Args:
            matrix (np.array of shape (4,4)): Homogenous Transformation Matrix
        """
        return Transform(translation=matrix[:3, 3], rotation=Rotation.from_matrix[:3, :3])

    def to_pos_quat(self):
        return self._translation, self._rotation.as_quat()
    
    def get_rotation(self):
        return self._rotation

    def get_translation(self):
        return self._translation

    def set_rotation(self, rotation):
        """
        Args:
            rotation (Rotation): new rotation of Homogenous Transform
        """
        self._rotation = rotation
        self._matrix[:3, :3] = rotation.as_matrix()
    
    def set_translation(self, translation):
        """
        Args:
            translation (np.array of shape (3): new translation of Homogenous Transform
        """
        self._translation = translation
        self._matrix[:3, 3] = translation
