import numpy as np
from roboticstoolbox import models

from utils import check_collision, check_edge

# Define robot and obstacles
robot = models.DH.Puma560()
q_min = [-175, -90, -150, -190, -120, -360]
q_max = [175, 85, 60, 190, 120, 360]

print(robot.q)
print(robot.qz)
print(robot.qn)
print(robot.qr)

colliding_sphere = robot.fkine(robot.q).t

link_radius = 0.05
sphere_centers = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.82]
])

sphere_radii = np.array([0.1, 0.1, 0.1])

# Generate random configuration
# q = np.random.uniform(q_min, q_max, size=6)

# Check for collisions
in_collision = check_collision(robot, robot.q, link_radius, sphere_centers, sphere_radii)
in_collision_edge = check_edge(robot, robot.q, robot.qz, link_radius, sphere_centers, sphere_radii)

if in_collision or in_collision_edge:
    print("Robot is in collision!")
else:
    print("Robot is collision-free.")
