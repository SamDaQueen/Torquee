import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import roboticstoolbox


# Define function to calculate torque cost
def torque_cost_for_edge(q, robot, qdot=np.zeros([7, 1]), qddot=np.zeros([7, 1])):
        tau = robot.rne(q, qdot, qddot)
        # Calculate torque cost as the sum of the squares of the joint torques
        cost = np.sum(np.square(tau))
        return cost


# Define PRM function to minimize torque cost
def prm_min_torque(q_start, q_goal, robot, samples=100, k=5, qdot=0, qddot=0):
    # Create a graph to hold the sampled nodes and edges
    G = nx.DiGraph()

    # Sample n points and add associated nodes
    for i in range(samples):
        # Generate random config
        rand_config = np.random.rand(7, 1)*2*np.pi
        rand_config[6] = 0
        # Potentially check collision
        # Add point to graph
        G.add_node(i, config=rand_config,
                   torque=torque_cost_for_edge(rand_config, robot))
    # Add start and end node
    G.add_node(samples, config=q_start, torque=torque_cost_for_edge(q_start, robot))
    G.add_node(samples+1, config=q_goal, torque=torque_cost_for_edge(q_goal, robot))

    # Add edges between k nearest neighbors for each node
    for i in range(samples+2):
        # Grab node config
        base_config = G.nodes[i]["config"]
        # Create empty dists matrix
        dists = np.empty([samples+2, 2])
        # For each node fine the k nearest nodes by config distance
        for j in range(samples+2):
            dists[j, 0] = j
            dists[j, 1] = np.sqrt(np.sum((G.nodes[j]["config"]-base_config)**2))
        # Sort by shortest distance
        # CHECK ME
        dists = np.sort(dists, axis=0)
        # Connect the closest cells and then add the torque of the ending node in the direction
        # of the connection to the edge to act as edge weight
        for j in range(1, k+1):
            G.add_edge(i, dists[j][0], torque=G.nodes[j]["torque"])
            G.add_edge(dists[j][0], i, torque=G.nodes[i]["torque"])

    # Find the shortest path
    path = nx.shortest_path(G, samples, samples+1, "torque")
    # Make the list of configs from the shortest path
    config_path = np.empty([len(path), 7])
    for i in range(len(path)):
        config = G.nodes[path[i]]
        config = config["config"]
        config_path[i] = config[0]

    # Return the list of configs
    return config_path


if __name__ == '__main__':
    robot = roboticstoolbox.models.DH.Panda()
    prm = prm_min_torque(np.zeros([7, 1]), np.ones([7, 1]), robot)
    print(prm)