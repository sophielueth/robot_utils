# Author: Sophie Lueth
# Date: June 2024

import numpy as np
from scipy.spatial.transform import Rotation


class Visualizer:
    """ Helper class for visualization in the gepetto-viewer
    """
    def __init__(self, gui, frame_diamter=0.01, frame_axis_length=0.1):
        """
        Args:
            gui(gepetto viwer): from pinocchio.RobotWrapper.BuildFromURDF.viewer.gui
        """
        
        self.gui = gui
        self.frame_diameter = frame_diamter
        self.frame_axis_length = frame_axis_length
        self.window_id = self.gui.getWindowID('python-pinocchio')
    
    def create_frame(self, name, alpha=0.7, radius=None, axis_length=None):
        """
        Args:
            name (str):
            alpha (float): transparency of frame (0 - not visible, 1 - not transparent)
            radius (float): in m
            axis_length(float): in m
        """
        if radius is None:
            radius = self.frame_diameter/2

        if axis_length is None:
            axis_length = self.frame_axis_length

        self.gui.addCylinder(f'world/{name}_x', radius, axis_length, (1., 0., 0., alpha)) # name, radius, height, color
        self.gui.addCylinder(f'world/{name}_y', radius, axis_length, (0., 1., 0., alpha))
        self.gui.addCylinder(f'world/{name}_z', radius, axis_length, (0., 0., 1., alpha))
    
    def draw_frame(self, name, translation=np.zeros(3,), rotation=Rotation([0, 0, 0, 1]), axis_length=None):
        """
        Args:
            name (str):
            translation (np.ndarray of shape 3):
            rotation (Rotation):
        """
        if axis_length is None:
            axis_length = self.frame_axis_length

        R_mat = rotation.as_matrix()
        t_x = translation + axis_length/2 * R_mat[:, 0]
        t_y = translation + axis_length/2 * R_mat[:, 1]
        t_z = translation + axis_length/2 * R_mat[:, 2]
        
        r_x = rotation * Rotation.from_euler('Y', np.pi/2)
        r_y = rotation * Rotation.from_euler('X', -np.pi/2)
        
        self.gui.applyConfiguration(f'world/{name}_x', t_x.tolist() + r_x.as_quat().tolist()) 
        self.gui.applyConfiguration(f'world/{name}_y', t_y.tolist() + r_y.as_quat().tolist()) 
        self.gui.applyConfiguration(f'world/{name}_z', t_z.tolist() + rotation.as_quat().tolist())
        
    def update(self):
        self.gui.refresh()
        self.gui.setCameraToBestFit(self.window_id)
        