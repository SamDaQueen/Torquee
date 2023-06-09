import time

import numpy as np
import pybullet as p
import pybullet_data
import roboticstoolbox as rtb

import utils
from robot_cspace import RobotCSpace
from src.genetic import GeneticAlgorithm


class Simulator:
    def __init__(self):
        """
        The __init__ function sets up the environment and robot, and initializes a few variables.
        """

        # Init Env Variables
        self.sphere_center = []
        self.sphere_radii = []
        self.dt = 0.05
        self.interp_time = 5
        self.wait_time = 2
        self.payload = 0
        self.robotId = 0
        self.physicsClient = 0

        # Useful inits
        zero_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_urdf_object = rtb.models.URDF.Puma560()
        robot_urdf = str(robot_urdf_object.urdf_filepath)
        robot_urdf = robot_urdf[0:len(robot_urdf) - 6]

        # Set up env
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        # Load robot
        self.robotId = p.loadURDF(robot_urdf, [0, 0, 0], zero_orientation, useFixedBase=True)

    # Objects is Nx4 containing [pybullet shape, x, y, z]
    def run(self, poses, sphere_centers=None, sphere_radii=None, dt=None, interp_time=None, wait_time=None, payload=0):
        """
        The run function runs the simulator.
        """
        self.dt = dt or self.dt
        self.interp_time = interp_time or self.interp_time
        self.wait_time = wait_time or self.wait_time
        self.payload = payload

        # Load Spheres
        if sphere_centers is not None:
            zero_orientation = p.getQuaternionFromEuler([0, 0, 0])

            # Load Plane
            p.loadURDF("plane.urdf", [0, 0, 0], zero_orientation)

            # Load Spheres
            ids = np.empty(len(sphere_centers))
            for i in range(len(ids)):
                ids[i] = p.loadURDF("sphere.urdf", sphere_centers[i],
                                    zero_orientation, globalScaling=sphere_radii[i])

        # sum_of_torques = 0

        # for previous, target in zip(self.poses[:-1], self.poses[1:]):
        #    sum_of_torques += self.robot.move_arm(target, self.dt, self.interp_time, self.wait_time, self.payload)

        time.sleep(10)

        # Move Robot Arm Section (poses = Nx6)
        cur_pose_goal = 0
        # Set first move
        joint_target = poses[cur_pose_goal, :]
        p.setJointMotorControlArray(self.robotId, range(6), controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_target)
        # Execute list of poses
        while cur_pose_goal < poses.shape[0]:
            # Get the current robot state of all joint
            cur_joint_state = p.getJointStates(self.robotId, range(6))
            # Extract only position states
            cur_pose = np.empty(6, )
            for i in range(6):
                cur_pose[i] = cur_joint_state[i][0]
            # Check if position state has reached goal
            math = np.abs(cur_pose - joint_target)
            compare = math < 0.01
            if np.all(compare):
                # Set robot target to next target
                if cur_pose_goal < poses.shape[0] - 1:
                    cur_pose_goal += 1
                joint_target = poses[cur_pose_goal, :] / 180 * 2 * np.pi
                p.setJointMotorControlArray(self.robotId, range(6), controlMode=p.POSITION_CONTROL,
                                            targetPositions=joint_target)
            # Step simulation
            p.stepSimulation()
            time.sleep(1. / 100.)

        # return sum_of_torques

    def reset(self):
        """
        The reset function resets the arm to the zero pose.
        """
        self.robot.reset()

    def close(self):
        """
        The close function closes the simulator.
        """
        p.disconnect(self.physicsClient)


if __name__ == '__main__':
    q_start = np.zeros([6])
    q_goal = np.transpose(np.array([85, 40, 40, 40, 40, 40]))
    target = np.array(np.deg2rad([85, 40, 40, 40, 40, 40]))
    joint_limits = np.deg2rad(utils.get_joint_limits())
    step_size = np.deg2rad(10)
    cspace = RobotCSpace(joint_limits, step_size)
    sphere_centers = utils.get_eval_sphere_centers()
    sphere_radii = utils.get_eval_sphere_radii()

    robot = rtb.models.DH.Puma560()

    start_time = time.time()
    # RRT
    # poses = RRT.RRT(q_start, q_goal, robot,
    #                 sphere_centers=utils.get_eval_sphere_centers(),
    #                 sphere_radii=utils.get_eval_sphere_radii())

    # PRM
    # poses = PRM.prm_min_torque(q_start, q_goal, robot, sphere_centers=utils.get_eval_sphere_centers(),
    #                            sphere_radii=utils.get_eval_sphere_radii())

    # A-star
    # path_cells = a_star_graph_search(robot, q_start, target, cspace, sphere_centers, sphere_radii)
    # poses = np.array([np.array(cspace.convert_cell_to_config(cell)) for cell in path_cells])
    # poses = np.rad2deg(poses)

    # poses = np.array([q_start, q_goal])

    # Greedy
    # path_cells = greedy(robot, q_start, target, cspace, sphere_centers, sphere_radii)
    # poses = np.array([np.array(cspace.convert_cell_to_config(cell)) for cell in path_cells])
    # poses = np.rad2deg(poses)

    # Genetic
    genetic = GeneticAlgorithm(robot, 10, 10, 0.6, 0.01, step_size=step_size,
                               sphere_radii=utils.get_eval_sphere_radii(),
                               sphere_centers=utils.get_eval_sphere_centers())
    path = genetic.run(q_start, target)
    poses = np.rad2deg(path)

    poses[0] = q_start

    end_time = time.time()

    distance, torque = utils.calculate_distance_torque(robot, np.deg2rad(poses))

    print("Time: ", end_time - start_time)
    print("Distance: ", distance)
    print("Torque: ", torque)
    if poses[0, 0] != np.inf:
        Simulator().run(poses, sphere_centers, sphere_radii)
    else:
        print("No solution found")
