import numpy as np


def check_collision(robot, link_radius, q, sphere_centers, sphere_radii, resolution=5):
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
