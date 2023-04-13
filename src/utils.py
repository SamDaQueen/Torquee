import numpy as np


def sample_spherical(coordinates, npoints, ndim=3, scale=0.05):
    coordinates = np.array(coordinates)
    coordinates = coordinates.reshape((len(coordinates), 1))
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= scale
    vec += coordinates
    return vec


def torque_cost(robot, q, qd, qdd):
    tau = robot.rne(q, qd, qdd)
    cost = np.sum(np.square(tau))
    return cost


def equal(q, qp, tolerance=1e-3):
    return np.allclose(q, qp, atol=tolerance)


def check_collision(robot, q, link_radius, sphere_centers, sphere_radii, resolution=11):
    x1 = np.zeros(3)
    T2 = robot.fkine(q)[:3, :3]
    x2 = T2.t
    T3 = T2 @ robot.fkine(q, 3)[:3, :3]
    x3 = T3.t

    if resolution is None:
        resolution = 11
    ticks = np.linspace(0, 1, resolution)  # 11 evenly spaced points between 0 and 1
    n = len(ticks)
    x12 = np.tile(x1, (n, 1)) + np.tile(x2 - x1, (n, 1)) * np.tile(ticks, (3, 1)).T
    x23 = np.tile(x2, (n, 1)) + np.tile(x3 - x2, (n, 1)) * np.tile(ticks, (3, 1)).T
    points = np.concatenate([x12, x23], axis=0)

    in_collision = False
    for i in range(sphere_centers.shape[0]):  # for each sphere
        if np.any(np.sum((points - np.tile(sphere_centers[i, :], (points.shape[0], 1)).T) ** 2, axis=0) < (
                link_radius + sphere_radii[i]) ** 2):
            in_collision = True
            break

    return in_collision

