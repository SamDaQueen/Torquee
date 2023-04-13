import numpy as np


class RobotCSpace:
    def __init__(self, joint_limits, step_size):
        self.joint_limits = joint_limits
        self.step_size = step_size
        self.num_cells = [int(np.ceil((limit[1] - limit[0]) / step_size)) + 1 for limit in joint_limits]
        self.grid = self.create_grid()

    def create_grid(self):

        grid = np.zeros(self.num_cells, dtype=bool)
        for i, limit in enumerate(self.joint_limits):
            # Calculate grid indices corresponding to joint limits
            lower_index = int(np.round((limit[0] % 360) / self.step_size))
            upper_index = int(np.round((limit[1] % 360) / self.step_size))

            # Set corresponding grid cells to True
            if upper_index >= lower_index:
                grid[(slice(None),) * i + tuple(range(lower_index, upper_index + 1))] = True
            else:
                grid[(slice(None),) * i + tuple(range(lower_index, self.num_cells[i]))] = True
                grid[(slice(None),) * i + tuple(range(upper_index + 1))] = True
        return grid

    def set_config_value(self, config, value):
        index = tuple(int(np.round((angle % 360) / step_size)) for angle in config)
        self.grid[index] = value

    def find_neighbors(self, config):
        neighbors = []
        for i in range(len(config)):
            for delta in [-1, 1]:
                neighbor_config = config.copy()
                neighbor_config[i] += delta * self.step_size
                if np.all(neighbor_config >= 0) and np.all(neighbor_config < 360):
                    index = tuple(int(np.round((angle % 360) / self.step_size)) for angle in neighbor_config)
                    if self.grid[index]:
                        neighbors.append(neighbor_config)
        return neighbors


if __name__ == '__main__':
    joint_limits = [(0, 360)] * 6
    step_size = 10
    config_space = RobotCSpace(joint_limits, step_size)

    joint_angles = [0, 40, 270, 350, 0, 0]
    valid = config_space.grid[tuple(int(np.round((angle % 360) / step_size)) for angle in joint_angles)]
    print(valid)

    neighbors = config_space.find_neighbors(np.array([0, 40, 270, 350, 0, 0]))
    print(neighbors)

    config_space.set_config_value([0, 40, 270, 350, 0, 0], False)

    valid = config_space.grid[tuple(int(np.round((angle % 360) / step_size)) for angle in joint_angles)]
    print(valid)
