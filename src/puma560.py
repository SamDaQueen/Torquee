import numpy as np
import roboticstoolbox as rtb


class Puma560:
    def __init__(self, env, orig_q=None):
        """
        The __init__ function sets up the robot and initializes a few variables.

        :param env: The simulator environment
        :param orig_q: Set the initial arm configuration
        """
        self.puma = rtb.models.Puma560()
        self.arm_configs = orig_q or self.puma.qz  # default to zero pose
        self.env = env
        self.puma.q = self.arm_configs
        self.env.add(self.puma, robot_alpha=True, collision_alpha=False)

    def move_arm(self, target, dt, interp_time, wait_time, payload=0):
        """
        The move_arm function moves the arm to the target pose.

        :param target: The target pose
        :param dt: The time step
        :param interp_time: The interpolation time
        :param wait_time: The wait time
        """
        traj = rtb.tools.trajectory.jtraj(self.puma.q, target, np.linspace(0.0, 1.0, int(interp_time / dt)))
        shape = traj.q.shape

        dh_puma = rtb.models.DH.Puma560()

        # Set payload
        dh_puma.payload(payload, np.r_[0, 0, 0.1])

        torques = []
        for i in range(shape[0]):
            self.puma.q = traj.q[i]
            torques.append(abs(dh_puma.rne(self.puma.q, traj.qd[i], traj.qdd[i])))

            self.env.step(dt)

        return np.sum(np.sum(torques, axis=1))

    def reset(self):
        """
        The reset function resets the arm to the zero pose.
        """
        self.move_arm(self.puma.qz, 0.01, 1, 1)
