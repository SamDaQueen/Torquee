import heapq
import numpy as np

from utils import distance_cost, torque_cost, PUMA_VELOCITY_LIMITS, PUMA_ACCELERATION_LIMITS


class PriorityQueue:

    def __init__(self):
        self.heap = []

    def push(self, item, priority=0):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        _, item = heapq.heappop(self.heap)
        return item

    def __len__(self):
        return len(self.heap)


def a_star_graph_search(robot, start, target, cspace, dt=1):
    frontier = PriorityQueue()

    start_cell = tuple(cspace.convert_config_to_cell(start))
    goal_cell = tuple(cspace.convert_config_to_cell(target))

    frontier.push(start_cell)

    seen = set()
    parent = {}

    distance = {start_cell: 0}
    velocities = {start_cell: 0}

    while frontier:
        current_cell = frontier.pop()

        if current_cell in seen:
            continue

        if current_cell == goal_cell:
            return reconstruct_path(parent, start_cell, current_cell)

        seen.add(current_cell)
        successors = cspace.find_neighbors(current_cell)

        current_config = np.array(cspace.convert_cell_to_config(current_cell))

        for successor_cell in successors:
            successor_config = np.array(cspace.convert_cell_to_config(successor_cell))

            if np.any(np.greater(np.abs(successor_config), robot.qlim[1, :])):
                continue

            qd = (successor_config - current_config) / dt
            qdd = (qd - velocities[tuple(current_cell)]) / dt

            qd_sign = np.sign(qd)
            qdd_sign = np.sign(qdd)

            qd = np.minimum(np.abs(qd), PUMA_VELOCITY_LIMITS) * qd_sign
            qdd = np.minimum(np.abs(qdd), PUMA_ACCELERATION_LIMITS) * qdd_sign

            # 90
            heuristic = 1.5 * (0.95 * distance_cost(robot, successor_config, target) +
                               0.05 * torque_cost(robot, successor_config, qd, qdd))

            cost = distance_cost(robot, successor_config, current_config) + 0
            # torque_cost(robot, successor_config, qdd, qdd)

            frontier.push(
                tuple(successor_cell),
                priority=distance[current_cell] + cost + heuristic
            )
            if (tuple(successor_cell) not in distance) \
                    or (distance[tuple(current_cell)] + cost < distance[tuple(successor_cell)]):
                distance[tuple(successor_cell)] = distance[tuple(current_cell)] + cost
                parent[tuple(successor_cell)] = current_cell
                velocities[tuple(successor_cell)] = qd

    return None


def reconstruct_path(came_from, start, end):
    """
    >>> came_from = {'b': 'a', 'c': 'a', 'd': 'c', 'e': 'd', 'f': 'd'}
    >>> reconstruct_path(came_from, 'a', 'e')
    ['a', 'c', 'd', 'e']
    """
    reverse_path = [end]
    while end != start:
        end = came_from[end]
        reverse_path.append(end)
    return list(reversed(reverse_path))
