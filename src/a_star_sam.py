import heapq

import numpy as np

from robot_cspace import RobotCSpace
from utils import torque_cost, equal


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, q, parent=None, g=0.0, h=0.0):
        """
        :param q: The configuration of the node
        :param parent: The parent of the node
        :param g: The cost to reach the node from the start node
        :param h: The heuristic cost to reach the goal node from the node
        """
        self.q = q
        self.parent = parent
        self.g = g
        self.h = h

    def f(self):
        """
        :return: The total cost of the node
        """
        return self.g + self.h

    def path(self):
        """
        :return: The path from the start node to the current node
        """
        path = [self.q]
        node = self.parent
        while node is not None:
            path.append(node.q)
            node = node.parent
        return list(reversed(path))


def distance(q1, q2):
    """
    Find the L2 distance between two configurations
    :param q1: configuration 1
    :param q2: configuration 2
    :return: L2 distance between the two configurations
    """
    return np.linalg.norm(np.array(q1) - np.array(q2))


def a_star(robot, q_start, q_goal):
    """
    Find a path from q_start to q_goal using A* search

    :param robot: The DHRobot object
    :param q_start: The start configuration
    :param q_goal: The goal configuration
    :return: The path from q_start to q_goal
    """
    joint_limits = list(zip(*robot.qlim))
    step_size = np.deg2rad(10)

    # joint_limits = [(0, 50)] * 6
    # step_size = 5

    cspace = RobotCSpace(joint_limits, step_size)

    if not cspace.is_valid(q_start):
        print("Start is not valid")
        return None

    if not cspace.is_valid(q_goal):
        print("Goal is not valid")
        return None

    open_list = []
    closed_list = []

    q_start = cspace.convert_config_to_cell(q_start)
    q_goal = cspace.convert_config_to_cell(q_goal)
    velocities = {tuple(q_start): 0}
    heapq.heappush(open_list, (0, Node(q_start)))

    dt = 0.1

    while open_list:
        _, current = heapq.heappop(open_list)

        if equal(current.q, q_goal):
            return current.path()

        closed_list.append(tuple(current.q))
        neighbors = cspace.find_neighbors(current.q)

        for q_next in neighbors:
            if tuple(q_next) in closed_list:
                continue

            g = current.g + distance(current.q, q_next)
            h = distance(cspace.convert_cell_to_config(q_next), cspace.convert_cell_to_config(q_goal))
            node = Node(q_next, current, g, h)

            # if check_collision(node.q, robot):
            #     continue

            qd = (q_next - current.q) / dt
            qdd = (qd - velocities[tuple(current.q)]) / dt

            if tuple(q_next) not in velocities:
                velocities[tuple(q_next)] = qd

            node_cost = torque_cost(robot, current.q, qd, qdd)
            node.f_cost = node.f() + node_cost

            heapq.heappush(open_list, (node.f_cost, node))

    return None
