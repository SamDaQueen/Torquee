import time

import numpy as np
import roboticstoolbox as rtb
import swift

from puma560 import Puma560
from robot_cspace import RobotCSpace
from src.greedy import greedy
from utils import get_joint_limits


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
    joint_limits = np.deg2rad(get_joint_limits())
    step_size = np.deg2rad(10)
    cspace = RobotCSpace(joint_limits, step_size)

    start = np.array([0, 0, 0, 0, 0, 0])
    target = np.array(np.deg2rad([85, 40, 40, 40, 40, 40]))

    start_time = time.time()
    # A-star
    # path_cells = a_star_graph_search(robot, start, target, cspace)

    # Greedy
    path_cells = greedy(robot, start, target, cspace)

    # Genetic
    # genetic = GeneticAlgorithm(robot, 10, 10, 0.6, 0.01, step_size=step_size)
    # path_cells = genetic.run(start, target)
    end_time = time.time()

    path = [np.array(cspace.convert_cell_to_config(cell)) for cell in path_cells]

    print(f'Time taken: {round(end_time - start_time, 2)} seconds.')
    print(path)

    simulator = Simulator(start)
    simulator.run(poses=path, dt=0.01, interp_time=1, wait_time=0.01)
