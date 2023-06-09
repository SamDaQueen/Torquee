import numpy as np

PUMA_TORQUE_LIMITS = np.array([97.6, 186.4, 89.4, 24.2, 20.1, 21.3])
PUMA_VELOCITY_LIMITS = np.array([8, 10, 10, 5, 5, 5])
PUMA_ACCELERATION_LIMITS = np.array([10, 12, 12, 8, 8, 8])


def sample_spherical(coordinates, npoints, ndim=3, scale=0.05):
    coordinates = np.array(coordinates)
    coordinates = coordinates.reshape((len(coordinates), 1))
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= scale
    vec += coordinates
    return vec


def torque(robot, q, qd, qdd):
    return robot.rne(q, qd, qdd)


def get_joint_limits():
    q_min = np.array([-85, -40, -40, -40, -40, -40])
    q_max = np.array([85, 40, 40, 40, 40, 40])
    return list(zip(q_min, q_max))


def rand_puma_config():
    q_min = np.array([-85, -40, -40, -40, -40, -40])
    q_max = np.array([85, 40, 40, 40, 40, 40])
    delta = q_max - q_min
    rand = np.random.rand(6) * delta
    config = rand + q_min
    return config


def torque_cost(robot, q, qd, qdd):
    tau = torque(robot, q, qd, qdd)
    cost = np.sqrt(np.sum(np.square(tau))) / np.sqrt(np.sum(np.square(PUMA_TORQUE_LIMITS)))
    return cost


def torque_cost_deg(q, robot, qd=np.zeros([6, 1]), qdd=np.zeros([6, 1])):
    q = np.deg2rad(q)
    qd = np.deg2rad(qd)
    qdd = np.deg2rad(qdd)
    tau = robot.rne(q, qd, qdd)
    cost = np.sum(np.square(tau))
    return cost


def equal(q, qp, tolerance=1e-3):
    return np.allclose(q, qp, atol=tolerance)


def check_collision(robot, q, sphere_centers, sphere_radii, link_radius=0.05, resolution=5):
    if len(sphere_centers) == 0:
        return False

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
                    # print("Collision detected at link {} position {} with sphere {}!".format(i, j, k))
                    in_collision = True
                    return in_collision

    return in_collision


def check_edge(robot, q_start, q_end, sphere_centers, sphere_radii, link_radius=0.05, resolution=5):
    if resolution is None:
        resolution = 11

    ticks = np.linspace(0, 1, resolution)
    n = len(ticks)

    # configs -> n configurations between q_start and q_end
    configs = np.tile(q_start, (n, 1)) + np.tile((np.array(q_end) - np.array(q_start)), (n, 1)) * np.tile(
        ticks.reshape(n, 1),
        (1, len(q_start)))

    in_collision = False
    for i in range(n):
        if check_collision(robot, configs[i, :], sphere_centers, sphere_radii, link_radius=link_radius):
            in_collision = True
            break

    return in_collision


def calculate_distance_torque(robot, path, dt=1):
    d = 0
    t = 0
    last_velocity = 0

    for i in range(1, len(path)):
        q_current = path[i]
        q_last = path[i - 1]
        current_velocity = (q_current - q_last) / dt
        current_acceleration = (current_velocity - last_velocity) / dt

        d += distance(q_current, q_last)
        t += torque(robot, q_current, current_velocity, current_acceleration)

        last_velocity = current_velocity

    return d, np.sum(t)


def get_eval_sphere_centers():
    sphere_centers = [
        [.7, .7, .2],
        [.5, -.5, .5],
        [.75, 0, .7],
        [-.75, 0, .5],
        # [-.25, .6, .2]
    ]
    return sphere_centers


def get_eval_sphere_radii():
    sphere_radii = [.3, .1, .1, .5]
    return sphere_radii


def distance_cost(robot, q1, q2):
    max_dist = np.linalg.norm(robot.qlim[1, :] - robot.qlim[0, :])
    return distance(q1, q2) / max_dist


def distance(q1, q2):
    """
    Find the L2 distance between two configurations
    :param q1: configuration 1
    :param q2: configuration 2
    :return: L2 distance between the two configurations
    """
    return np.linalg.norm(np.array(q1) - np.array(q2))
