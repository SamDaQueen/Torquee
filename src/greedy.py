import math

import numpy as np

from utils import torque, torque_cost, distance, distance_cost, \
    equal, PUMA_TORQUE_LIMITS, \
    PUMA_ACCELERATION_LIMITS, PUMA_VELOCITY_LIMITS, check_collision, check_edge


class GreedyNode:
    """
    A node class for Greedy Pathfinding
    """

    def __init__(self, q, parent=None, h=math.inf):
        """
        :param q: The configuration of the node
        :param parent: The parent of the node
        :param h: The heuristic cost to reach the goal node from the node
        """
        self.q = q
        self.parent = parent
        self.h = h

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

    def __lt__(self, other):
        """
        :return: True if the total cost of the node is less than the other node
        """
        return self.h < other.h

    def __str__(self):
        return f'[{self.q} - {self.h}]'


def greedy(robot, q_start, q_goal, cspace, sphere_centers, sphere_radii, dt=1):
    """
    Find a path from q_start to q_goal using A* search

    :param cspace: The configuration space
    :param sphere_radii:  The radii of the spheres
    :param sphere_centers:  The centers of the spheres
    :param robot: The DHRobot object
    :param q_start: The start configuration
    :param q_goal: The goal configuration
    :return: The path from q_start to q_goal
    """

    if not cspace.is_valid(q_start):
        print("Start is not valid")
        return None

    if not cspace.is_valid(q_goal):
        print("Goal is not valid")
        return None

    closed_list = set()

    q_start = cspace.convert_config_to_cell(q_start)
    q_goal = cspace.convert_config_to_cell(q_goal)
    velocities = {tuple(q_start): 0}

    current = GreedyNode(q_start)

    while current is not None and not equal(current.q, q_goal):
        closed_list.add(tuple(current.q))
        neighbors = cspace.find_neighbors(current.q)
        best_node, best_heuristic = None, math.inf

        distances = np.array([distance(cspace.convert_cell_to_config(q_next),
                                       cspace.convert_cell_to_config(q_goal)) for q_next in neighbors])
        candidate_indices = np.argsort(distances)[:50]
        candidates = [neighbors[i] for i in candidate_indices]

        current_config = np.array(cspace.convert_cell_to_config(current.q))

        for q_next in candidates:
            if tuple(q_next) in closed_list:
                continue

            q_next_config = np.array(cspace.convert_cell_to_config(q_next))

            if check_collision(robot, np.rad2deg(q_next_config), sphere_centers, sphere_radii) or \
                    check_edge(robot, np.rad2deg(current_config), np.rad2deg(q_next_config), sphere_centers, sphere_radii):
                continue

            if np.any(np.greater(np.abs(q_next_config), robot.qlim[1, :])):
                continue

            qd = (q_next_config - current_config) / dt
            qdd = (qd - velocities[tuple(current.q)]) / dt

            qd_sign = np.sign(qd)
            qdd_sign = np.sign(qdd)

            qd = np.minimum(np.abs(qd), PUMA_VELOCITY_LIMITS) * qd_sign
            qdd = np.minimum(np.abs(qdd), PUMA_ACCELERATION_LIMITS) * qdd_sign

            if np.any(np.greater_equal(
                    np.abs(torque(robot, q_next_config, qd, qdd)),
                    PUMA_TORQUE_LIMITS)):
                continue

            if tuple(q_next) not in velocities:
                velocities[tuple(q_next)] = qd

            # h = .9 * distance_cost(robot, cspace.convert_cell_to_config(q_next),
            #                        cspace.convert_cell_to_config(q_goal)) + \
            #     .1 * torque_cost(robot, current.q, qd, qdd)
            h = distance_cost(robot, q_next_config, cspace.convert_cell_to_config(q_goal)) + \
                torque_cost(robot, q_next_config, qd, qdd)
            # h = torque_cost(robot, q_next_config, qd, qdd)

            node = GreedyNode(q_next, current, h)

            # if check_collision(node.q, robot):
            #     continue

            if best_heuristic > node.h:
                best_node, best_heuristic = node, node.h

        current = best_node

    return current.path() if current is not None else None
