import numpy as np


class RobotCSpace:
    def __init__(self, joint_limits, step_size):
        """
        Initializes the configuration space of the robot.

        :param joint_limits: The joint limits of the robot
        :param step_size: The step size of the configuration space
        """
        self.joint_limits = joint_limits
        self.step_size = step_size
        self.num_cells = [int(np.ceil((limit[1] - limit[0]) / step_size)) + 1 for limit in joint_limits]
        self.grid = np.ones(self.num_cells, dtype=bool)

    def get_grid(self):
        """
        Returns the grid of the configuration space

        :return: the grid
        """
        return self.grid

    def set_config_value(self, config, value):
        """
        Sets the value of a configuration in the grid

        :param config: The configuration in joint angles
        :param value: True or False
        """
        index = self.convert_config_to_cell(config)
        # return if config is invalid
        if not self.is_valid(config):
            return
        self.grid[index] = value

    def convert_config_to_cell(self, config):
        """
        Converts a configuration to a cell index in the grid

        :param config:  The configuration in joint angles
        :return: The cell index
        """
        # Convert cell index to 6D coordinate
        return [int((coord - limit[0]) / self.step_size) for coord, limit in zip(config, self.joint_limits)]

    def convert_cell_to_config(self, cell):
        """
        Converts a cell index in the grid to a configuration in joint angles

        :param cell:    The cell index
        :return:    The configuration in joint angles
        """
        # Convert 6D coordinate to cell index
        return [coord * self.step_size + limit[0] for coord, limit in zip(cell, self.joint_limits)]

    def find_neighbors(self, config):
        """
        Finds the neighbors of a configuration in the grid

        :param config:  The configuration in joint angles
        :return:    A list of neighbors in joint angles
        """
        # return if config is invalid
        if not self.is_valid(config):
            return []

        coord = self.convert_config_to_cell(config)

        # generate list of possible neighbor coordinates
        valid_neighbors = []
        for i in range(6):
            for offset in [-1, 0, 1]:
                neighbor_index = tuple(coord[j] + (offset if i == j else 0) for j in range(6))

                if neighbor_index != tuple(coord) and all(
                        0 <= neighbor_index[j] < self.num_cells[j] for j in range(6)) and \
                        self.grid[neighbor_index]:
                    valid_neighbors.append(neighbor_index)

        # convert neighbor coordinates to joint angles
        valid_neighbors = [self.convert_cell_to_config(neighbor) for neighbor in valid_neighbors]

        return valid_neighbors

    def is_valid(self, config):
        """
        Checks if a configuration is valid

        :param config:  The configuration in joint angles
        :return:    True if valid, False otherwise
        """
        index = self.convert_config_to_cell(config)
        return all(0 <= i < n for i, n in zip(index, self.grid.shape))


if __name__ == '__main__':
    q_min = [-175, -90, -150, -190, -120, -360]
    q_max = [175, 85, 60, 190, 120, 360]
    # joint_limits = [(0, 30)] * 6
    joint_limits = list(zip(q_min, q_max))

    step_size = 10
    config_space = RobotCSpace(joint_limits, step_size)

    neighbors = config_space.find_neighbors(np.array([0, 0, 0, 0, 0, 0]))
    print(neighbors)
