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

    def find_neighbors(self, coord):
        """
        Finds the neighbors of a configuration in the grid

        :param coord:  The configuration in grid cell coordinates
        :return:    A list of neighbors in joint angles
        """
        # return if config is invalid
        if not self.is_valid(self.convert_cell_to_config(coord)):
            raise ValueError("Invalid input")

        lower = self.convert_config_to_cell([t[0] for t in self.joint_limits])
        upper = self.convert_config_to_cell([t[1] for t in self.joint_limits])

        # generate list of possible neighbor coordinates
        valid_neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    for da in [-1, 0, 1]:
                        for db in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dx == dy == dz == da == db == dc == 0:
                                    continue
                                neighbor_coord = [coord[0] + dx, coord[1] + dy, coord[2] + dz, coord[3] + da,
                                                  coord[4] + db, coord[5] + dc]
                                for j in range(6):
                                    if neighbor_coord[j] < lower[j]:
                                        neighbor_coord[j] += upper[j] - lower[j]
                                    elif neighbor_coord[j] > upper[j]:
                                        neighbor_coord[j] -= upper[j] - lower[j]
                                if self.is_valid(self.convert_cell_to_config(tuple(neighbor_coord))):
                                    valid_neighbors.append(neighbor_coord)

        return np.array(valid_neighbors)

    def is_valid(self, config):
        """
        Checks if a configuration is valid

        :param config:  The configuration in joint angles
        :return:    True if valid, False otherwise
        """
        return all(self.joint_limits[i][0] <= config[i] <= self.joint_limits[i][1] for i in range(len(
            self.joint_limits)))


if __name__ == '__main__':
    # joint_limits = list(zip(*rtb.models.DH.Puma560().qlim))
    # print(joint_limits)
    # step_size = np.deg2rad(10)

    joint_limits = [(-30, 30)] * 6
    step_size = 5

    config_space = RobotCSpace(joint_limits, step_size)

    neighbors = config_space.find_neighbors(np.array([5, 0, 2, 1, 4, 6]))
    print(neighbors)
