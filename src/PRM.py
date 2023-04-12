import networkx as nx
import numpy as np

# Define function to calculate torque cost
def torque_cost_for_edge(q, qdot, qddot, robot):
        tau = robot.rne(q, qdot, qddot)
        # Calculate torque cost as the sum of the squares of the joint torques
        cost = np.sum(np.square(tau))
        return cost

# Define PRM function to minimize torque cost
def prm_min_torque(q_start, q_goal, robot, samples=1000, k=5, qdot=0, qddot=0, ):
        # Create a graph to hold the sampled nodes and edges
        G = nx.Graph()
        # Create an array to store the configs of the nodes
        configs = np.empty((samples, 7))

        # Sample n points
        for i in range(samples):
            # Generate random config
            rand_vec = np.random.rand(6)
            rand_config = rand_vec*360
            # Potentially check collision
            # Add point to graph
            G.add_node(rand_vec)
            configs[i] = [i, rand_config]

if __name__ == '__main__':
    prm = prm_min_torque()