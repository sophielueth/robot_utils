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
    return J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))


class PinWrapper:
    """
    Wrapper for pinocchio with ROS (if wanted), offers Forward & Inverse Kinematics & Dynamics & Jacoby-Matrices

    from robot_utils.pinocchio.pinocchio_wrapper import PinWrapper
    """

    def __init__(self, model=None, urdf_path=None, robot_description_param='/robot_description',
                 tool_frame='tool_frame',
                 joints_of_interest=None, q0_pin=None):
        """
        Args:
            model (pin.Model): pinocchio model, if already generated some other way; if given, urdf_path & robot_description_param will be ignored
            urdf_path (string): path to urdf file to be loaded; if given, robot_description_param will be ignored
            robot_description_param (string): the topic on the ROS param server to read out the robot description
            tool_frame (string): name of frame to consider tool_frame/default frame for FK & IK; has to be in self.model.frames
            joints_of_interest (list of strings): if given, reduced model with these joints will be built; have to be in self.model.names; requires q0_pin to be set
            q0_pin (np.array of shape (model.nq,)): only needed if joints_to_lock set; configuration to lock non-relevant joints; e.g. from reference config in a srdf file
        """
        # =============================================== CREATE MODEL =============================================== #
        if model is not None:
            self.model_full = model
        elif urdf_path is not None:
            self.model_full = pin.buildModelFromUrdf(urdf_path)
        else:
            import rospy
            robot_description = rospy.get_param(robot_description_param)
            self.model_full = pin.buildModelFromXML(robot_description)

        if joints_of_interest is not None:  # list not empty
            if q0_pin is None:
                print('[PinWrapper Error]: If you set joints_of_interest to build a reduced pinocchio model, you also '
                      'need to set q0_pin to define the joint configuration for the rest of the joints.')
                exit(0)

            # identify joints to lock
            self.q_actuated = [False] * self.model_full.nq
            joints_to_lock = [True] * self.model_full.njoints

            for joint in joints_of_interest:
                if not self.model_full.existJointName(joint):
                    print('[PinWrapper WARNING]: joint ' + joint + ' does not belong to the model! Check for typos...')
                else:
                    joint_id = self.model_full.getJointId(joint)
                    joints_to_lock[joint_id] = False
                    joint_idx_q = self.model_full.idx_qs[joint_id]
                    self.q_actuated[joint_idx_q] = True
                    if self.model_full.nqs[joint_id] == 2:
                        self.q_actuated[joint_idx_q + 1] = True

            joint_ids_to_lock = []
            for joint in np.array(self.model_full.names)[joints_to_lock][1:]:
                joint_ids_to_lock.append(self.model_full.getJointId(joint))

            self.model = pin.buildReducedModel(self.model_full, joint_ids_to_lock, reference_configuration=q0_pin)
        else:
            self.model = self.model_full
            self.q_actuated = [True] * self.model.nv

        # =============================================== SETUP DATA ================================================ #
        self.data = self.model.createData()  # for algorithmic buffering
        self.ee_frame_name = tool_frame
        assert self.model.existFrame(self.ee_frame_name)

        self.joints_unlimited = np.array(self.model.nqs[1:]) == 2
        self.joints_pin_indices = self.model.idx_qs[1:]

        self.q_last_fk = None  # to skip repeated computations
        self.q_last_jac = None  # to skip repeated computations

    def get_frame_id(self, frame_name):
        """
        Returns the pinocchio frame id. Raises Value Error if frame not in model

        Args:
            frame_name (str): name of the frame
        """
        frame_id = self.model.getFrameId(frame_name)
        if not frame_id < self.model.nframes or frame_id is None:
            raise ValueError("Unknown frame " + frame_name)

        return frame_id

    def get_joint_id(self, joint_name):
        """
        Returns the pinocchio joint id. Raises Value Error if frame not in model

        Args:
            joint_name (str): name of the joint
        """
        joint_id = self.model.getJointId(joint_name)
        if not joint_id < self.model.njoints or joint_id is None:
            raise ValueError("Unknown joint " + joint_name)

        return joint_id

    def forward_kinematics(self, q, frame=None, frame_id=None, pin_conf=False, pin_se3=True):
        # TODO: To be tested
        """Computes the homogenous transform at the specified joint for the given joint configuration.

        Args:
            q (np.ndarray of shape (model.nv,) or (11,)): joint configuration to compute forward kinematics for in rad
            frame (str): name of the frame to compute the transform for
            frame_id (int): pinocchio frame id of the frame to compute the transform for
            pin_conf (bool): boolean indicating whether jont configuration given in pinocchio format (True) or not (False)
            pin_se3 (bool): boolean indicating whether to return result as pin.SE3 (True) or Transform (False)

        Return:
            pin.SE3/Transform: homogenous transform at the end-effector, type depends on pin_se3 parameter
        """
        if frame_id is not None and frame is not None:
            print('Please only enter frame name OR frame_id into the forward kinematics function!')
            exit(0)
        
        if frame_id is None:
            if frame is None:
                frame = self.ee_frame_name
            frame_id = self.model.getFrameId(frame)

        if not pin_conf:
            q_pin = self.to_q_pin(q)
        else:
            q_pin = q
        if not np.all(q_pin == self.q_last_fk):
            self.q_last_fk = q_pin
            pin.framesForwardKinematics(self.model, self.data, q_pin)

        transform = self.data.oMf[int(frame_id)]
        if pin_se3:
            return transform
        else:
            return Transform(translation=transform.translation, rotation=Rotation.from_matrix(transform.rotation))

    def inverse_kinematics(self, des_trans, q=None, frame=None, pos_threshold=0.005, angle_threshold=5. * np.pi / 180,
                           n_trials=7, dt=0.1):
        """Get IK joint configuration for desired pose of specified joint frame.

        Args:
            des_trans (Transform): desired frame transform for the frame specified via joint_ind
            q (np.ndarray of shape (model.nv,)): joint start configuration in rad, if applicable
            frame (str): name of the frame that is des_trans 
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
            frame = self.ee_frame_name

        oMdes = pin.SE3(des_trans.to_matrix())
        frame_id = self.model.getFrameId(frame)

        if q is not None:
            q_pin = self.to_q_pin(q=q)

        for n in range(n_trials):
            if q is None:
                q_pin = np.random.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit)

            for i in range(800):
                pin.framesForwardKinematics(self.model, self.data, q_pin)
                oMf = self.data.oMf[frame_id]
                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector

                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q_pin, frame_id)
                v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q_pin = pin.integrate(self.model, q_pin, v * dt)
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

    def jacobian(self, q, frame=None, frame_id=None, pin_conf=False):
        # TODO: test
        """
        computes Jacobian for given joint configuration for a given Frame; optimized for frequent calls with same q

        Args:
            q (np.ndarray of shape (model.nv,)): joint configuration
            frame (str): The frame to compute the Jacobian for
            frame_id (int): pinocchio frame id of the frame to compute the transform for
            pin_conf (bool): boolean indicating whether jont configuration given in pinocchio format (True) or not (False)

        Returns:
            np.ndarray of shape (6,model.nv)
        """
        if frame_id is not None and frame is not None:
            print('Please only enter frame name OR frame_id into the jacobian function!')
            exit(0)

        if frame_id is None:
            if frame is None:
                frame = self.ee_frame_name
            frame_id = self.model.getFrameId(frame)
        if not pin_conf:
            q_pin = self.to_q_pin(q)
        else:
            q_pin = q

        if not np.all(q_pin == self.q_last_jac):
            self.q_last_jac = q_pin
            pin.computeJointJacobians(self.model, self.data, q_pin)

        jac = pin.getFrameJacobian(self.model, self.data, int(frame_id), pin.WORLD)

        return jac

    def jacobian_dot(self, q, q_dot, frame=None):
        # TODO: test (dimensions?)
        """returns dJ/dt, with J being the Jacobian Matrix

        Args:
            q (np.ndarray of shape (model.nv,)):
            q_dot (np.ndarray of shape (model.nv,)):
            frame (str): frame to compute jacobian_dot for; if None, takes End-effector

        Returns:
            np.ndarray of shape (6, model.nv): the time derivative of the Jacobian
        """
        if frame is None:
            frame = self.ee_frame_name
        frame_id = self.model.getFrameId(frame)

        q_pin, q_dot_pin = self.to_q_pin(q, q_dot)

        pin.computeJointJacobiansTimeVariation(self.model, self.data, q_pin, q_dot_pin)
        return pin.getFrameJacobianTimeVariation(self.model, self.data, frame_id, pin.WORLD)

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
