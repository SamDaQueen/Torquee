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
    G.add_node(0, cost=utils.torque_cost_deg(q_start, robot), q=q_start, qd=0)
    configs = np.array([q_start])

    i = 0
    while i < max_samples:
        print(i)
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
        if dists[0, 1] < step_size:
            q_new = q_rand
        else:
            q_new = q_near + ((step_size / (np.linalg.norm(q_rand - q_near))) * (q_rand - q_near))
        # Check if q_new is free
        if not utils.check_collision(robot, q_new, sphere_centers, sphere_radii):
            # Check for lowest cost node in neighborhood to connect to new node
            lowest_cost = np.inf
            lowest_cost_node = -1
            for i in range(G.number_of_nodes()):
                if dists[i, 1] < neighborhood and not utils.check_edge(
                        robot, q_new, G.nodes[dists[i, 0]]["q"], sphere_centers, sphere_radii):
                    q_old = G.nodes[dists[i, 0]]["q"]
                    qd_old = G.nodes[dists[i, 0]]["qd"]
                    qd_new = (q_new - q_old) / dt
                    qdd_new = (qd_new - qd_old) / dt
                    new_node_cost = utils.torque_cost_deg(q_new, robot, qd_new, qdd_new)
                    old_node_cost = G.nodes[dists[i, 0]]["cost"]
                    if new_node_cost > old_node_cost:
                        cost = new_node_cost
                    else:
                        cost = old_node_cost
                    if cost < lowest_cost:
                        lowest_cost = cost
                        lowest_cost_node = dists[i, 0]
            if lowest_cost_node != -1:
                q_old = G.nodes[lowest_cost_node]["q"]
                qd_new = (q_new - q_old) / dt
                # Add node
                G.add_node(G.number_of_nodes(), config=q_new, q=q_new, qd=qd_new)
                G.add_edge(lowest_cost_node, G.number_of_nodes(), cost=lowest_cost)
                configs = np.concatenate((configs, [q_new]), axis=0)
            # Check if new nodes in neighborhood have new node connect to it
            for i in range(G.number_of_nodes() - 1):
                if dists[i, 1] < neighborhood and not utils.check_edge(
                        robot, q_new, G.nodes[dists[i, 0]]["q"], sphere_centers, sphere_radii):
                    end_current_cost = G.nodes[dists[i, 0]]["cost"]
                    q_old = G.nodes[dists[i, 1]]["q"]
                    qd_old = G.nodes[dists[i, 1]]["qd"]
                    qd_new = (q_old - q_new) / dt
                    qdd_new = (qd_old - qd_new) / dt
                    new_edge_cost = utils.torque_cost_deg(q_old, robot, qd_new, qdd_new)
                    start_current_cost = G.nodes[G.number_of_nodes() - 1]["cost"]

                    if end_current_cost < new_edge_cost or end_current_cost < start_current_cost:
                        G.add_edge(G.number_of_nodes() - 1, dists[i, 0], cost=np.max([
                            new_edge_cost, start_current_cost]))
            # Check if new node is within the q_goal radius
            if np.sqrt(np.sum((q_goal - q_new) ** 2, axis=0)) < goal_radius:
                return graph_to_path(G, q_goal)
        i += 1
    nx.draw_networkx(G)
    plt.show()
    return np.ones([6, 1]) * np.Inf


def graph_to_path(G, q_goal):
    # Find the shortest path
    path = nx.shortest_path(G, 0, G.number_of_nodes() - 1, "torque")
    # Make the list of configs from the shortest path
    config_path = np.zeros([len(path) + 1, 6])
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
    rrt = RRTS(np.zeros([6]), np.transpose(np.array([175, 85, 60, 190, 120, 360])),
               robot)  # , sphere_centers=sphere_centers, sphere_radii=sphere_radii)
    print(rrt)
