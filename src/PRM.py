import networkx as nx
import numpy as np
import roboticstoolbox
import utils


# Define PRM function to minimize torque cost
def prm_min_torque(q_start, q_goal, robot, samples=500, k=5, sphere_centers=[], sphere_radii=[]):
    # Create a graph to hold the sampled nodes and edges
    dt = 1
    G = nx.DiGraph()
    configs = np.empty([samples+2, 6])

    # Sample n points and add associated nodes
    i = 0
    while i < samples:
        # Generate random config
        rand_config = utils.rand_puma_config()
        # Check collision of config
        if not utils.check_collision(robot, rand_config, sphere_centers, sphere_radii):
            # Add node to graph
            configs[i, :] = rand_config
            G.add_node(i, q=rand_config)
            i += 1
    # Add start and end node
    G.add_node(samples, q=q_start, qd=0)
    G.add_node(samples+1, q=q_goal, qd=0)
    configs[samples, :] = q_start
    configs[samples + 1, :] = q_goal

    print("Done adding nodes")

    # Add edges between k nearest neighbors for each node
    for i in range(samples+2):
        # Grab node config
        base_config = G.nodes[i]["q"]
        # Create empty dists array
        dists = np.empty([samples+2, 2])
        # For each node fine the k nearest nodes by config distance
        dists[:, 0] = np.arange(samples+2)
        dists[:, 1] = np.sqrt(np.sum((configs-base_config)**2, axis=1))
        # Sort by shortest distance
        dists = dists[dists[:, 1].argsort()]
        # Connect the closest cells and then add the torque of the ending node in the direction
        # of the connection to the edge to act as edge weight
        j = 1
        num_neighbors = k
        while j < k:
            j_config = G.nodes[dists[j][0]]["q"]
            # Check edge and make sure its valid, if not valid check the next edge
            if utils.check_edge(robot, base_config, j_config, sphere_centers, sphere_radii):
                num_neighbors += 1
            else:
                # Add the edge
                G.add_edge(i, dists[j][0], torque=utils.torque_cost_deg(j_config, robot,
                                                                        (j_config-base_config)/dt))
                G.add_edge(dists[j][0], i, torque=utils.torque_cost_deg(base_config, robot,
                                                                        (base_config-j_config)/dt))
            j += 1

    print("Done adding edges")

    # Find the shortest path
    path = nx.shortest_path(G, samples, samples+1, "torque")
    # Make the list of configs from the shortest path
    config_path = np.empty([len(path), 6])
    for i in range(len(path)):
        config = G.nodes[path[i]]
        config = config["q"]
        config_path[i] = config

    # Return the list of configs
    return config_path


if __name__ == '__main__':
    robot = roboticstoolbox.models.DH.Puma560()
    sphere_centers = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
    ])
    sphere_radii = np.array([0.1, 0.1])
    prm = prm_min_torque(np.zeros([6]), np.transpose(np.array([85, 40, 40, 40, 40, 40])),
                         robot)
    print(prm)