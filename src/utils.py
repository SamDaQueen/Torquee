import numpy as np


def sample_spherical(coordinates, npoints, ndim=3, scale=0.05):
    coordinates = np.array(coordinates)
    coordinates = coordinates.reshape((len(coordinates), 1))
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= scale
    vec += coordinates
    return vec


def rand_puma_config():
    q_min = np.array([[-175, -90, -150, -190, -120, -360]])
    q_max = np.array([[175, 85, 60, 190, 120, 360]])
    delta = q_max[0] - q_min[0]
    rand = np.random.rand(6) * delta
    config = rand + q_min
    return np.transpose(config)


def torque_cost(robot, q, qd, qdd):
    tau = robot.rne(q, qd, qdd)
    cost = np.sum(np.sqrt(np.square(tau))) / len(q)
    return cost


def torque_cost_prm(q, robot, qd=np.zeros([6, 1]), qdd=np.zeros([6, 1])):
    q = np.deg2rad(q)
    qd = np.deg2rad(qd)
    qdd = np.deg2rad(qdd)
    tau = robot.rne(q, qd, qdd)
    cost = np.sum(np.square(tau))
    return cost


def equal(q, qp, tolerance=1e-3):
    return np.allclose(q, qp, atol=tolerance)


def check_collision(robot, q, sphere_centers, sphere_radii, link_radius=0.05, resolution=5):
    links = robot.links
    in_collision = False
    fkine = robot.fkine_all(q)

    # Define the positions of interest for each link
    positions = [np.linspace(fkine[i].t, fkine[i + 1].t, resolution) for i in [0, 1, 2, 3]]

    # Check for collisions at each position
    for i, link_pos in enumerate(positions):
        for j, pos in enumerate(link_pos):
            for k, sphere_center in enumerate(sphere_centers):
                dist = np.linalg.norm(pos - sphere_center)
                if dist <= (link_radius + sphere_radii[k]):
                    print("Collision detected at link {} position {} with sphere {}!".format(i, j, k))
                    in_collision = True
                    return in_collision

    return in_collision


def check_edge(robot, q_start, q_end, sphere_centers, sphere_radii, link_radius=0.05, resolution=5):
    if resolution is None:
        resolution = 11

    ticks = np.linspace(0, 1, resolution)
    n = len(ticks)
    q_start = q_start[:, 0]
    q_end = q_end[:, 0]

    # configs -> n configurations between q_start and q_end
    configs = np.tile(q_start, (n, 1)) + np.tile((q_end - q_start), (n, 1)) * np.tile(ticks.reshape(n, 1),
                                                                                      (1, len(q_start)))

    in_collision = False
    for i in range(n):
        if check_collision(robot, configs[i, :], sphere_centers, sphere_radii, link_radius=link_radius):
            in_collision = True
            break

    return in_collision
