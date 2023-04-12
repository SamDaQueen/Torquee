import spatialgeometry as sg
import swift
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import pybullet as pb


from puma560 import Puma560


class Simulator:
    def __init__(self):
        """
        The __init__ function sets up the environment and robot, and initializes a few variables.
        """
        self.env = swift.Swift()
        physicsClient = pb.connect(pb.GUI)
        self.env.launch(realtime=True)
        self.robot = Puma560(self.env)
        self.dt = 0.05
        self.interp_time = 5
        self.wait_time = 2
        self.poses = [self.robot.puma.qz, self.robot.puma.qr, self.robot.puma.qs,
                      self.robot.puma.qd, self.robot.puma.qz]
        self.payload = 0
        self.objects = [[sg.Sphere(1), 1, 1, 1]]

    # Objects is Nx4 containing [pybullet shape, x, y, z]
    def run(self, poses=None, dt=None, interp_time=None, wait_time=None, payload=0, objects=None):
        """
        The run function runs the simulator.
        """
        self.dt = dt or self.dt
        self.interp_time = interp_time or self.interp_time
        self.wait_time = wait_time or self.wait_time
        self.poses = poses or self.poses
        self.payload = payload
        self.objects =  objects or self.objects

        sphere = pb.createCollisionShape(pb.GEOM_SPHERE)
        self.env.add(sphere)

        sum_of_torques = 0

        for previous, target in zip(self.poses[:-1], self.poses[1:]):
            sum_of_torques += self.robot.move_arm(target, self.dt, self.interp_time, self.wait_time, self.payload)

        self.robot.reset()

        return sum_of_torques

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
    simulator = Simulator()
    sum_of_torques = simulator.run(dt=0.1, interp_time=1, wait_time=0.01, payload=1)
    print(sum_of_torques)
