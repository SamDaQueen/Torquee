#!/usr/bin/env python
import swift
import roboticstoolbox as rtb
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

# Create a puma in the default zero pose
puma = rtb.models.Puma560()

pumaDH = rtb.models.DH.Puma560()
# puma.qz = np.array( [ 2.6486,     -2.38986263,  2.98768081,  0.76815354,  0.86615253,  2.5455596 ])
puma.q = puma.qz

print(f'Puma: {puma}')

env.add(puma, robot_alpha=True, collision_alpha=False)

dt = 0.05
interp_time = 5
wait_time = 2

# Pass through the reference poses one by one.
# This ignores the robot collisions, and may pass through itself
poses = [puma.qz, puma.rd, puma.ru, puma.lu, puma.ld]
for previous, target in zip(poses[:-1], poses[1:]):
    for alpha in np.linspace(0.0, 1.0, int(interp_time / dt)):
        puma.q = previous + alpha * (target - previous)
        print(f'Puma.q: {pumaDH.itorque(puma.q, np.ones((6,)))}')
        env.step(dt)
    for _ in range(int(wait_time / dt)):
        puma.q = target
        env.step(dt)

# Uncomment to stop the browser tab from closing
env.hold()
