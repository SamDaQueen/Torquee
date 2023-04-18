import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import utils
import roboticstoolbox as rtb


def RRTS(q_start, q_goal, robot, neighborhood=100, max_samples=2000, goal_radius=50, step_size=10,
         bias=0.1, sphere_centers=[], sphere_radii=[]):
    # Create a graph to hold the sampled nodes and edges
    dt = 1
    G = nx.DiGraph()
    G.add_node(0, config=q_start)
    configs = np.array([q_start])

    i = 0
    while i < max_samples:
        # Generate random config or bias toward goal
        if np.random.rand() < bias:
            q_rand = q_goal
        else:
            q_rand = utils.rand_puma_config()
        # Find the nearest node in graph to new node
        # Find distance to each node
        dists = np.zeros([configs.shape[0], 2])
        dists[:, 0] = np.arange(configs.shape[0])
        dists[:, 1] = np.sqrt(np.sum((configs - q_rand) ** 2, axis=1))
        # Sort by shortest distance
        dists = dists[dists[:, 1].argsort()]
        # Create new node config
        q_near = configs[int(dists[0, 0])]
        if dists[0, 1] < 10:
            q_new = q_rand
        else:
            q_new = q_near + ((step_size/(np.linalg.norm(q_rand-q_near)))*(q_rand-q_near))
        # Check if q_new is free and edge is free
        if not utils.check_collision(robot, q_new, sphere_centers, sphere_radii) and not utils.check_edge(robot, q_near, q_new, sphere_centers, sphere_radii):
            # Add node and edge if found a valid edge
            G.add_node(len(configs), config=q_new)
            G.add_edge(dists[0, 0], len(configs), torque=utils.torque_cost_prm(
                q_new, robot, (q_near - q_new) / dt))
            configs = np.concatenate((configs, [q_new]), axis=0)
            # Check if new node is within the q_goal radius
            if np.sqrt(np.sum((q_goal - q_new) ** 2, axis=0)) < goal_radius:
                return graph_to_path(G, q_goal)
        i += 1
    nx.draw_networkx(G)
    plt.show()
    return np.ones([6, 1])*np.Inf


def graph_to_path(G, q_goal):
    # Find the shortest path
    path = nx.shortest_path(G, 0, G.number_of_nodes()-1, "torque")
    # Make the list of configs from the shortest path
    config_path = np.zeros([len(path)+1, 6])
    for i in range(len(path)):
        config = G.nodes[path[i]]
        config = config["config"]
        config_path[i, :] = config
    # Add goal config
    config_path[len(path), :] = q_goal
    # Return the list of configs
    return config_path


if __name__ == '__main__':
    robot = rtb.models.DH.Puma560()
    sphere_centers = np.array([
        [.5, .5, .5]
    ])
    sphere_radii = np.array([.1])
    rrt = RRT(np.zeros([6]), np.transpose(np.array([175, 85, 60, 190, 120, 360])),
              robot, sphere_centers=sphere_centers, sphere_radii=sphere_radii)
    print(rrt)
