import numpy as np

from greedy import distance_cost
from utils import torque_cost, PUMA_VELOCITY_LIMITS, PUMA_ACCELERATION_LIMITS, torque, PUMA_TORQUE_LIMITS, equal, \
    check_collision, check_edge


class GeneticAlgorithm:
    def __init__(self, robot, population_size, num_generations, crossover_rate, mutation_rate, sphere_centers,
                 sphere_radii, dt=0.1, step_size=0.1):
        self.velocities = None
        self.robot = robot
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.joint_limits = list(zip(*robot.qlim))
        self.dt = dt
        self.step_size = step_size
        self.sphere_centers = sphere_centers
        self.sphere_radii = sphere_radii

    def run(self, q_start, q_goal):

        # Initialize population
        population = self._initialize_population(q_start, q_goal)

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}/{self.num_generations}...")
            # Evaluate fitness of population

            fitness_scores = [self.fitness_function(individual) for individual in population]

            # Select parents for crossover
            parents = self.selection_method(population, fitness_scores)

            # Perform crossover to generate new offspring
            offspring = []
            for i in range(self.population_size):
                parent1, parent2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
                child = self._crossover(parent1, parent2, self.crossover_rate)
                offspring.append(child)

            # Perform mutation on offspring
            for i in range(self.population_size):
                offspring[i] = self._mutate(offspring[i], self.mutation_rate, q_goal)

            # Evaluate fitness of offspring
            offspring_fitness_scores = [self.fitness_function(individual) for individual in offspring]

            # Merge population and offspring
            combined_population = population + offspring

            # Select the best individuals to form the next generation
            population = self.selection_method(combined_population, fitness_scores + offspring_fitness_scores)
            self.velocities = {tuple(q_start): 0}

        # Return the best solution found
        return population[0][0]

    def _create_candidate(self, q_current, q_goal, last_vel):
        beta = 0.1

        while True:
            # Move towards q_goal with probability beta
            if np.random.rand() < beta:
                q_next = self.step_size * (q_goal - q_current) + q_current
                qd, qdd = self._calculate_qd_qdd(q_next, q_current, last_vel)
                return q_next, qd, qdd
            q_next = np.zeros_like(q_current, dtype=np.float64)
            for i in range(len(q_current)):
                q_range = self.joint_limits[i]
                q_min = max(q_range[0], q_current[i] - self.step_size)
                q_max = min(q_range[1], q_current[i] + self.step_size)
                val = np.random.uniform(q_min, q_max)
                q_next[i] = val
            qd, qdd = self._calculate_qd_qdd(q_next, q_current, last_vel)
            if not check_collision(self.robot, np.rad2deg(q_next), self.sphere_centers, self.sphere_radii) or \
                    check_edge(self.robot, np.rad2deg(q_current),
                               np.rad2deg(q_next), self.sphere_centers, self.sphere_radii):
                break
        return q_next, qd, qdd

    def _initialize_population(self, q_start, q_goal):
        # Generate a population of random paths from q_start to q_goal
        population = []
        for i in range(self.population_size):
            print(f"Initializing population {i + 1}/{self.population_size}...")
            velocities = [0]
            accelerations = [0]
            path = [q_start]
            while not equal(path[-1], q_goal):
                q, qd, qdd = self._create_candidate(path[-1], q_goal, velocities[-1])  # Generate a random configuration
                if self.evaluation_function(path[-1], q, qd, qdd) \
                        < self.evaluation_function(path[-1], q_goal, qd, qdd):
                    velocities.append(qd)
                    accelerations.append(qdd)
                    path.append(q)
                if len(path) > 2000:
                    break
            if not equal(path[-1], q_goal):
                qd, qdd = self._calculate_qd_qdd(q_goal, path[-1], velocities[-1])
                velocities.append(qd)
                accelerations.append(qdd)
                path.append(q_goal)
            population.append((path, velocities, accelerations))
        return population

    def _crossover(self, parent1, parent2, crossover_rate):
        if np.random.rand() > crossover_rate:
            return parent1

        # Uniform crossover
        child_q = []
        min_length = min(len(parent1[0]), len(parent2[0]))
        for i in range(min_length):
            if np.random.rand() < 0.5:
                child_q.append(parent1[0][i])
            else:
                child_q.append(parent2[0][i])

        # Append the remaining configurations from the longer parent
        if len(parent1[0]) > len(parent2[0]):
            child_q.extend(parent1[0][min_length:])
        else:
            child_q.extend(parent2[0][min_length:])

        # Consider the step size when generating the child
        for i in range(1, len(child_q)):
            diff = np.abs(child_q[i] - child_q[i - 1])
            if np.any(diff > self.step_size):
                child_q[i] = np.sign(child_q[i] - child_q[i - 1]) * self.step_size + child_q[i - 1]

        child_velocities = [0]
        child_accelerations = [0]

        for i in range(1, len(child_q)):
            qd, qdd = self._calculate_qd_qdd(child_q[i], child_q[i - 1], child_velocities[i - 1])
            child_velocities.append(qd)
            child_accelerations.append(qdd)

        return child_q, child_velocities, child_accelerations

    def _mutate(self, individual, mutation_rate, q_goal):
        mutated_q = [individual[0][0]]

        velocities = [0]
        accelerations = [0]
        for i in range(1, len(individual[0]) - 1):
            if np.random.rand() < mutation_rate / 1.2:
                continue
            if np.random.rand() < mutation_rate:
                q, qd, qdd = self._create_candidate(mutated_q[-1], q_goal,
                                                    velocities[-1])  # Generate a new random configuration
                mutated_q.append(q)
                velocities.append(qd)
                accelerations.append(qdd)
            else:
                qd, qdd = self._calculate_qd_qdd(individual[0][i], mutated_q[-1], velocities[-1])
                mutated_q.append(individual[0][i])
                velocities.append(qd)
                accelerations.append(qdd)
        return mutated_q, velocities, accelerations

    def selection_method(self, population, fitness_scores):
        fitness_scores = np.array(fitness_scores)
        # Replace negative fitness scores with the minimum fitness score
        min_fitness = np.min(fitness_scores[np.logical_not(np.isneginf(fitness_scores))])
        fitness_scores = np.where(np.isneginf(fitness_scores), min_fitness, fitness_scores)
        total_fitness = np.sum(fitness_scores)
        probabilities = fitness_scores / total_fitness
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        selected_individuals = [population[i] for i in selected_indices]
        return selected_individuals

    def _calculate_qd_qdd(self, q_next, q_current, last_vel):
        qd = (q_next - q_current) / self.dt
        qdd = (qd - last_vel) / self.dt

        qd_sign = np.sign(qd)
        qdd_sign = np.sign(qdd)

        qd = np.minimum(np.abs(qd), PUMA_VELOCITY_LIMITS) * qd_sign
        qdd = np.minimum(np.abs(qdd), PUMA_ACCELERATION_LIMITS) * qdd_sign

        return qd, qdd

    def evaluation_function(self, q_current, q_next, qd, qdd):
        if np.any(np.greater_equal(
                np.abs(torque(self.robot, q_next, qd, qdd)),
                PUMA_TORQUE_LIMITS)):
            return np.inf

        return 90 * distance_cost(self.robot, q_current, q_next) + 10 * torque_cost(self.robot, q_current, qd, qdd)

    def fitness_function(self, individual):

        total_cost = 0

        path, velocities, accelerations = individual
        for i in range(len(path) - 1):
            q_current = path[i]
            q_next = path[i + 1]
            velocity = velocities[i + 1]
            acceleration = accelerations[i + 1]
            cost = self.evaluation_function(q_current, q_next, velocity, acceleration)
            total_cost += cost

            if check_collision(self.robot, np.rad2deg(q_next), self.sphere_centers, self.sphere_radii) or \
                    check_edge(self.robot, np.rad2deg(q_current), np.rad2deg(q_next), self.sphere_centers,
                               self.sphere_radii):
                continue

        return 1 / total_cost
