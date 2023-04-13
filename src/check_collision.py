import numpy as np


def check_collision(robot, q, link_radius, sphere_centers, sphere_radii, resolution=11):
    x1 = np.zeros(3)
    T2 = robot.fkine(q)[:3, :3]  # homogenous transform
    x2 = T2.t  # The translation
    T3 = T2 @ robot.fkine(q, 3)[:3, :3]
    x3 = T3.t

    if resolution is None:
        resolution = 11
    ticks = np.linspace(0, 1, resolution)  # 11 evenly spaced points between 0 and 1
    n = len(ticks)
    # x12 -> n points on straight line from origin to joint 3
    x12 = np.tile(x1, (n, 1)) + np.tile(x2 - x1, (n, 1)) * np.tile(ticks, (3, 1)).T
    # x13 -> n points on straight line on last arm
    x23 = np.tile(x2, (n, 1)) + np.tile(x3 - x2, (n, 1)) * np.tile(ticks, (3, 1)).T
    # all the points
    points = np.concatenate([x12, x23], axis=0)

    in_collision = False
    for i in range(sphere_centers.shape[0]):  # for each sphere
        # Find L2 distance from each point found above with center of
        # sphere.
        # check if any distance > sum od radii of sphere and link
        if np.any(np.sum((points - np.tile(sphere_centers[i, :], (points.shape[0], 1)).T) ** 2, axis=0) < (
                link_radius + sphere_radii[i]) ** 2):
            in_collision = True
            break

    return in_collision
