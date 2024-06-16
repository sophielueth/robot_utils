# Author: Sophie Lueth
# Date: June 2024

import numpy as np
import pinocchio as pin

from scipy.spatial.transform import Rotation
from robot_utils.spatial.transform import Transform


def damped_pseudo_inverse(J, damp=1e-10):
    """
   Returns damped pseudo inverse of J, according to:

      min_{q_dot} (x_dot - J @ q_dot).T @ (x_dot - J @ q_dot) + damp * q_dot.T * q_dot
      q_dot = J.T @ (J @ J.T + damp*I).inv x_dot

   Is numerically more stable than the pseudo inverse in case of singularities.

   Args:
      J (np.ndarray of shape (6, q_dim)): robotic Jacobian matrix according to: x_dot = J @ q
      damp (float): damping coefficient
   Returns:
      np.ndarray of shape (q_dim, 6): damped_pseudo_inverse
   """
    return J.T @ np.linalg.inv(J @ J.T + damp*np.eye(6))


class PinWrapper:
    """
    Wrapper for pinocchio with ROS (if wanted), offers Forward & Inverse Kinematics & Dynamics & Jacoby-Matrices

    from robot_utils.pinocchio.pinocchio_wrapper import PinWrapper
    """
    def __init__(self, robot_description_param='/robot_description', urdf_path=None, tool_frame='tool_frame'):
        """
        Args:
            robot_description_param (string): the topic on the ROS param server to read out the robot description
            urdf_path (string): path to urdf file to be loaded; if this parameter is active, the input of robot_description_patam will be ignored
            tool_frame (string): name of frame to consider tool_frame/default frame for FK & IK; has to be in self.model.frames
        """
        if urdf_path is None:
            import rospy
            robot_description = rospy.get_param(robot_description_param)
            self.model = pin.buildModelFromXML(robot_description)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path)

        self.data = self.model.createData() # for algorithmic buffering
        self.ee_frame = tool_frame

        self.joints_unlimited = []
        self.joints_pin_indices = []
        for joint in self.model.joints:
            if self.model.njoints > joint.id >= 0:
                self.joints_pin_indices.append(joint.idx_q)
                self.joints_unlimited.append(joint.nq == 2)

    def forward_kinematics(self, q, frame=None):
        """Computes the homogenous transform at the specified joint for the given joint configuration.

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration to compute forward kinematics for in rad
            frame (str): name of the frame to compute the transform for 

        Return:
            Transform: homogenous transform at the end-effector
        """
        q_pin = self.to_q_pin(q=q)
        
        if frame is None:
            frame = self.ee_frame
        
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        
        frame_id = self.model.getFrameId(frame)
        frame = self.data.oMf[frame_id]
        return Transform(translation=frame.translation, rotation=Rotation.from_matrix(frame.rotation))

    def inverse_kinematics(self, des_trans, q=None, frame=None, pos_threshold=0.005, angle_threshold=5.*np.pi/180, n_trials=7, dt=0.1):
        """Get IK joint configuration for desired pose of specified joint frame.

        Args:
            des_trans (Transform): desired frame transform for the frame specified via joint_ind
            q (np.ndarray of shape (model.nv,)): joint start configuration in rad, if applicable
            frame (str): name of the frame to compute the inverse kinematics for
            pos_threshold (float): in m 
            angle_threshold (float): in rad
            n_trials (int):
            dt (float): in s, used as stepsize for gradient descent (Jacobian)

        Return:
            bool, : whether the inverse kinematics found a solution within the
                    thresholds
            np.ndarray of shape (model.nv,) : best joint configuration found/the
                    first one to fulfill the requirement thresholds
        """
        damp = 1e-10
        success = False
        
        if frame is None:
            frame = self.ee_frame
        
        oMdes = pin.SE3(des_trans.to_matrix())
        frame_id = self.model.getFrameId(frame)
        
        if q is not None:
            q_pin = self.to_q_pin(q=q)

        for n in range(n_trials):
            if q is None:
                q_pin = np.random.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit)

            for i in range (800):
                pin.framesForwardKinematics(self.model, self.data, q_pin)
                oMf = self.data.oMf[frame_id]
                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector
                
                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q_pin, frame_id)
                v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q_pin = pin.integrate(self.model, q_pin, v*dt)
                q_pin = np.clip(q_pin, self.model.lowerPositionLimit, self.model.upperPositionLimit)

            if success:
                best_q_pin = np.array(q_pin)
                break
            else:
                # Save current solution
                best_q_pin = np.array(q_pin)

        best_q = self.from_q_pin(best_q_pin)

        return success, best_q
    
    def forward_dynamics(self, q, q_dot, tau):
        """Computing joint accelerations for given joint configuration, joint velocity and joint torque.

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration in rad
            q_dot (np.ndarray of shape (model.nv,)): joint velocity in rad/s
            tau (np.ndarray of shape (model.nv,)): joint torque in Nm

        Return:
            np.ndarray of shape (model.nv,) : joint acceleration in rad/s^2
        """

        q_pin, q_dot_pin = self.to_q_pin(q=q, q_dot=q_dot)

        q_dotdot = pin.aba(self.model, self.data, q_pin, q_dot_pin, tau)
        # TODO: add selection for relevant joints
        # q_dotdot = q_dotdot[:7] 
        
        return q_dotdot

    def inverse_dynamics(self, q, q_dot=None, q_dotdot=None):
        """Computing the necessary joint torques to achieve the given acceleration
        for joint configuration and velocity.

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration in rad
            q_dot (np.ndarray of shape (model.nv,)): joint velocity in rad/s
            q_dotdot (np.ndarray of shape (model.nv,)): joint acceleration in rad/s^s

        Returns:
            np.ndarray of shape (model.nv,): joint torque in Nm
        """
        if q_dot is None: q_dot = np.zeros((self.model.nv,))
        if q_dotdot is None: q_dotdot = np.zeros((self.model.nv,))

        q_pin, q_dot_pin, q_dotdot_pin = self.to_q_pin(q=q, q_dot=q_dot, q_dotdot=q_dotdot)

        tau = pin.rnea(self.model, self.data, q_pin, q_dot_pin, q_dotdot_pin)

        return tau

    def mass_matrix(self, q):
        """returns the 7x7 mass matrix for the given joint configuration

        Args:
            q (np.ndarray of shape (self.model.nv,)): joint configuration in rad
        Returns:
            np.ndarray of shape (self.model.nv, self.model.nv): mass matrix for the gripperless robot
        """
        q_pin = self.to_q_pin(q=q)

        pin.crba(self.model, self.data, q_pin)
        # TODO: add selection for relevant joints: maybe over reduced model with defined relevant joints?
        # return self.data.M[:7, :7]
        return self.data.M

    def coriolis_vector(self, q, q_dot):
        """computes coriolis vector for the given joint configuration & velocity

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration in rad
            q_dot (np.ndarray of shape (model.nv,)): joint velocity configuration in rad/s

        Returns:
            np.ndarray of shape (model.nv,): coriolis vector for the gripperless robot
        """
        q_pin, q_dot_pin = self.to_q_pin(q=q, q_dot=q_dot)

        pin.computeCoriolisMatrix(self.model, self.data, q_pin, q_dot_pin)
        # TODO: add selection for relevant joints: maybe over reduced model with defined relevant joints?
        # return self.data.C[:7, :7] @ q_dot
        return self.data.C @ q_dot

    def gravity_vector(self, q):
        """computes torques needed to compensate gravity

        Args:
            q (np.ndarray of shape (model.nv,): joint configuration in rad

        Returns: np.ndarray of shape (model.nv,): grav comp vector

        """
        q_pin = self.to_q_pin(q=q)
        pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        return self.data.g

    def jacobian(self, q, frame=None):
        # TODO: test dimension?
        """computes Jacobian for given joint configuration

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration
            frame (str): name of frame to compute jacobian for, default ee

        Returns:
            np.ndarray of shape (6, model.nv)
        """
        q_pin = self.to_q_pin(q=q)

        if frame is None:
            frame = self.ee_frame
        frame_id = self.model.getFrameId(frame)
        # TODO: add selection for relevant joints: maybe over reduced model with defined relevant joints?
        # return pin.computeFrameJacobian(self.model, self.data, q_pin, frame_id)[:6, :7]
        return pin.computeFrameJacobian(self.model, self.data, q_pin, frame_id)[:6, :]

    def jacobian_dot(self, q, q_dot):
        # TODO: test dimension?
        """returns dJ/dt, with J being the Jacobian Matrix

        Args:
            q (np.ndarray of shape (model.nv,)):
            q_dot (np.ndarray of shape (model.nv,)):

        Returns:
            np.ndarray of shape (6, model.nv): the time derivative of the Jacobian
        """
        q_pin, q_dot_pin = self.to_q_pin(q, q_dot)

        pin.computeJointJacobiansTimeVariation(self.model, self.data, q_pin, q_dot_pin)
        # TODO: add selection for relevant joints: maybe over reduced model with defined relevant joints?
        # return pin.getFrameJacobianTimeVariation(self.model, self.data, 7, pin.WORLD)[:7, :7]
        return pin.getFrameJacobianTimeVariation(self.model, self.data, 7, pin.WORLD)

    def to_q_pin(self, q=None, q_dot=None, q_dotdot=None):
        # TODO: test
        """transforms given (model.nv,) shape np.ndarrays to pinocchio internal compatible shape

        Args:
            q (np.ndarray of shape (model.nv,)): in rad
            q_dot (np.ndarray of shape (model.nv,)):
            q_dotdot (np.ndarray of shape (model.nv,))

        Return:
            q_pin (np.ndarray of shape (model.nq, 1)): has unlimited rotational joints as np.cos(q_i) and np.sin(q_i)
            q_dot_pin (np.ndarray of shape (model.nv, 1))
            q_dotdot_pin (np.ndarray of shape (model.nv, 1))
        """
        res = []

        if q is not None:
            q_pin = []
            for q_i, joint_unlimited in zip(q, self.joints_unlimited):
                if joint_unlimited:
                    q_pin.append(np.cos(q_i))
                    q_pin.append(np.sin(q_i))
                else:
                    q_pin.append(q_i)

            res.append(np.array(q_pin).reshape(-1, 1))

        if q_dot is not None:
            res.append(q_dot.reshape(-1, 1))

        if q_dotdot is not None:
            res.append(q_dotdot.reshape(-1, 1))

        if len(res) > 1:
            return tuple(res)
        elif len(res) == 1:
            return res[0]

    def from_q_pin(self, q_pin):
        # TODO: test
        """transforms given joint configuration (model.nq,) shape np.ndarray from pinocchio internal compatible shape to human-readable (model.nv) shape

        Args:
            q_pin (np.ndarray of shape (model.nq,)): has unlimited rotational joints as np.cos(q_i) and np.sin(q_i)

        Return:
            q (np.ndarray of shape (model.nv, )): in rad
        """
        q = []

        for joint_unlimited, joint_pin_idx in zip(self.joints_unlimited, self.joints_pin_indices):
            if joint_unlimited:
                q.append(np.arctan2(q_pin[joint_pin_idx + 1], q_pin[joint_pin_idx]))
            else:
                q.append(q_pin[joint_pin_idx])

        return np.array(q)
    