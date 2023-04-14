import pybullet as p
import pybullet_data
import time

class Simulator:
    def __init__(self):
        """
        The __init__ function sets up the environment and robot, and initializes a few variables.
        """

        # Init Env Variables
        self.dt = 0.05
        self.interp_time = 5
        self.wait_time = 2
        self.poses = []
        self.payload = 0
        self.objects = [[]]

        # Set up env
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        # Load objects
        zero_orientation = p.getQuaternionFromEuler([0, 0, 0])
        planeId = p.loadURDF("plane.urdf")
        #sphereId = p.loadURDF("sphere.urdf", [0, .5, .5], zero_orientation, globalScaling=10)
        sphereVisualId = p.createVisualShape(p.GEOM_SPHERE)
        sphereId = p.createMultiBody(baseVisualShapeIndex=sphereVisualId, basePosition=[0, .5, 0.5])
        robotId = p.loadURDF("\\franka_panda\panda.urdf", [0, 0, 0], zero_orientation, useFixedBase=True) #flags=p.URDF_USE_SELF_COLLISION

        # Test robot arm move
        p.setJointMotorControlArray(robotId, range(6), controlMode=p.POSITION_CONTROL, targetPositions=[1.5]*6)

        robotAABB = p.getAABB(robotId)
        sphereAABB = p.getAABB(sphereId)

        # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
        for i in range(1000):
            p.stepSimulation()
            p.performCollisionDetection()
            time.sleep(1. / 500.)
            contacts = p.getOverlappingObjects(sphereAABB[0], sphereAABB[1])
            print(len(contacts))
            #contacts = p.getContactPoints()
            #print("CONTACTED: " + str(contacts[0][0]))

        p.disconnect()


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
        self.objects = objects or self.objects

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
