import numpy as np
import roboticstoolbox as rtb
import swift

from puma560 import Puma560
from a_star_sam import a_star
from greedy import greedy
from robot_cspace import RobotCSpace


class Simulator:
    def __init__(self, start):
        """
        The __init__ function sets up the environment and robot, and initializes a few variables.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.robot = Puma560(self.env, start)
        self.dt = 0.05
        self.interp_time = 5
        self.wait_time = 2
        self.poses = [self.robot.puma.qz, self.robot.puma.qr, self.robot.puma.qs,
                      self.robot.puma.qd, self.robot.puma.qz]

    def run(self, poses=None, dt=None, interp_time=None, wait_time=None):
        """
        The run function runs the simulator.
        """
        self.dt = dt or self.dt
        self.interp_time = interp_time or self.interp_time
        self.wait_time = wait_time or self.wait_time
        self.poses = poses or self.poses

        for previous, target in zip(self.poses[:-1], self.poses[1:]):
            self.robot.move_arm(target, self.dt, self.interp_time, self.wait_time)

        self.robot.reset()

    def reset(self):
        """
        The reset function resets the arm to the zero pose.
        """
        self.robot.reset()

    def close(self):
        """
        The close function closes the simulator.
        """
        self.env.close()


if __name__ == '__main__':
    robot = rtb.models.DH.Puma560()

    # search_class = AStarSearch(robot, [])
    # path = search_class.run(start, np.array([0, 30, 20, 10, 0, 0]))

    # Joint limits: [[-2.7925268  -1.91986218 -2.35619449 -4.64257581 -1.74532925 -4.64257581],
    #               [ 2.7925268   1.91986218  2.35619449  4.64257581  1.74532925  4.64257581]]
    # path = a_star(robot, np.array([0, 0, 0, -1, 0, 0]), np.array([0, 0, 0, -1, 0, 1]))
    joint_limits = list(zip(*robot.qlim))
    step_size = np.deg2rad(10)
    cspace = RobotCSpace(joint_limits, step_size)

    start = np.array([0, 0, 0, -1, 0, 0])
    target = np.array([2.6486, -1.80, -2.1416, 0.6743, 0.8604, 2.6611])
    path_cells = greedy(robot, start, target, cspace)
    path = [np.array(cspace.convert_cell_to_config(cell)) for cell in path_cells]

    print(path)

    simulator = Simulator(start)
    simulator.run(poses=path, dt=0.01, interp_time=1, wait_time=1)
