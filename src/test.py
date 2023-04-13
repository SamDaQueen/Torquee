import numpy as np
from roboticstoolbox import models

from check_collision import check_collision

# Define robot and obstacles
robot = models.DH.Puma560()
q_min = [-175, -90, -150, -190, -120, -360]
q_max = [175, 85, 60, 190, 120, 360]

colliding_sphere = robot.fkine(robot.q).t

link_radius = 0.05
sphere_centers = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5]
    colliding_sphere
])

sphere_radii = np.array([0.1, 0.1, 0.1, 0.1])

# Generate random configuration
# q = np.random.uniform(q_min, q_max, size=6)

# Check for collisions
in_collision = check_collision(robot, link_radius, robot.q, sphere_centers, sphere_radii)

if in_collision:
    print("Robot is in collision!")
else:
    print("Robot is collision-free.")
