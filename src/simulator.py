import swift

from src.puma560 import Puma560

<<<<<<< HEAD
# Create a puma in the default zero pose
puma = rtb.models.Puma560()

pumaDH = rtb.models.DH.Puma560()
# puma.qz = np.array( [ 2.6486,     -2.38986263,  2.98768081,  0.76815354,  0.86615253,  2.5455596 ])
puma.q = puma.qz

print(f'Puma: {puma}')

env.add(puma, robot_alpha=True, collision_alpha=False)
=======
>>>>>>> 84e19fc0063bec5e18b1f5d904e538970d1a8c9f

class Simulator:
    def __init__(self):
        """
        The __init__ function sets up the environment and robot, and initializes a few variables.
        """
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.robot = Puma560(self.env)
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
    simulator = Simulator()
    simulator.run(dt=0.1, interp_time=1, wait_time=0.01)
