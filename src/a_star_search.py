import utils
from robot_cspace import RobotCSpace
import heapq
import numpy as np


class AStarSearch:
    def __init__(self, robot, obstacles):
        self.robot = robot
        self.obstacles = obstacles

    def heuristic(self, q, qd, qddot, target):
        return utils.torque_cost(self.robot, q, qd, qddot) + np.linalg.norm(target - q) / len(q)

    def cost(self, start, q):
        return np.linalg.norm(q - start)

    def run(self, target):
        start = self.robot.q

        frontier = []
        heapq.heappush(frontier, (0, start))

        # Initialize the visited set
        visited = set()

        # Initialize the parent dictionary
        parent = {tuple(start): None}
        velocities = {tuple(start): 0}

        joint_limits = [(0, 360)] * 6
        step_size = 10

        dt = 1

        cspace = RobotCSpace(joint_limits, step_size)

        # Start the search
        while len(frontier) > 0:
            # Get the current node with the minimum cost
            q_current = heapq.heappop(frontier)[1]

            if tuple(q_current) in visited:
                continue

            # Check if we reached the goal
            if utils.equal(q_current, target):
                # Reconstruct the path
                path = [q_current]
                while not utils.equal(q_current, start):
                    q_current = parent[tuple(q_current)]
                    path.append(q_current)
                path.reverse()
                return path

            # Add the current node to the visited set
            visited.add(tuple(q_current))

            q_neighbors = cspace.find_neighbors(q_current)

            for q_neighbor in q_neighbors:
                qd = (q_neighbor - q_current) / dt
                qdd = (qd - velocities[tuple(q_current)]) / dt
                heapq.heappush(frontier, (
                    self.cost(start, q_neighbor) + self.heuristic(q_neighbor, qd, qdd, target),
                    q_neighbor
                ))

                if tuple(q_neighbor) not in parent:
                    parent[tuple(q_neighbor)] = q_current

                if tuple(q_neighbor) not in velocities:
                    velocities[tuple(q_neighbor)] = qd

        # If we reached here, the search failed
        return None
