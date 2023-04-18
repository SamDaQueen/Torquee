import numpy as np

from greedy import distance_cost
from utils import torque_cost, PUMA_VELOCITY_LIMITS, PUMA_ACCELERATION_LIMITS, torque, PUMA_TORQUE_LIMITS, equal


class GeneticAlgorithm:
    def __init__(self, robot, population_size, num_generations, crossover_rate, mutation_rate, dt=0.1, step_size=0.1):
        self.velocities = None
        self.robot = robot
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.joint_limits = list(zip(*robot.qlim))
        self.dt = dt
        self.step_size = step_size
        self.fitness_scores = None

    def run(self, q_start, q_goal):

        self.velocities = {tuple(q_start): 0}

        # Initialize population
        population = self._initialize_population(q_start, q_goal)

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}/{self.num_generations}...")
            # Evaluate fitness of population
            self.fitness_scores = [self.fitness_function(individual) for individual in population]

            # Select parents for crossover
            parents = self.selection_method(population, self.fitness_scores)

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
            population = self.selection_method(combined_population, self.fitness_scores + offspring_fitness_scores)
            self.velocities = {tuple(q_start): 0}

        # Return the best solution found
        return population[0]

    def _create_candidate(self, q_current, q_goal):
        beta = 0.1
        # Move towards q_goal with probability beta
        if np.random.rand() < beta:
            q_next = self.step_size * (q_goal - q_current) + q_current
            return q_next
        q_next = np.zeros_like(q_current, dtype=np.float64)
        for i in range(len(q_current)):
            q_range = self.joint_limits[i]
            q_min = max(q_range[0], q_current[i] - self.step_size)
            q_max = min(q_range[1], q_current[i] + self.step_size)
            val = np.random.uniform(q_min, q_max)
            q_next[i] = val
        return q_next

    def _initialize_population(self, q_start, q_goal):
        # Generate a population of random paths from q_start to q_goal
        population = []
        for i in range(self.population_size):
            print(f"Initializing population {i + 1}/{self.population_size}...")
            path = [q_start]
            while not equal(path[-1], q_goal):
                candidate = self._create_candidate(path[-1], q_goal)  # Generate a random configuration
                if self.evaluation_function(path[-1], candidate) < self.evaluation_function(path[-1], q_goal):
                    path.append(candidate)
                if len(path) > 1000:
                    break
            if not equal(path[-1], q_goal):
                path.append(q_goal)
            population.append(path)
        return population

    def _crossover(self, parent1, parent2, crossover_rate):
        if np.random.rand() > crossover_rate:
            return parent1

        # Uniform crossover
        child = []
        min_length = min(len(parent1), len(parent2))
        for i in range(min_length):
            if np.random.rand() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

        # Append the remaining configurations from the longer parent
        if len(parent1) > len(parent2):
            child.extend(parent1[min_length:])
        else:
            child.extend(parent2[min_length:])

        # Consider the step size when generating the child
        child = np.array(child)
        for i in range(1, len(child)):
            diff = np.abs(child[i] - child[i - 1])
            if np.any(diff > self.step_size):
                child[i] = np.sign(child[i:] - child[i - 1:]) * self.step_size + child[i - 1]
            if tuple(child[i]) not in self.velocities:
                self.velocities[tuple(child[i])] = (child[i] - child[i - 1]) / self.dt

        return child

    def _mutate(self, individual, mutation_rate, q_goal):
        for i in range(1, len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = self._create_candidate(individual[i - 1], q_goal)  # Generate a new random configuration
                if tuple(individual[i]) not in self.velocities:
                    self.velocities[tuple(individual[i])] = (individual[i] - individual[i - 1]) / self.dt
        return individual

    def selection_method(self, population, fitness_scores):
        fitness_scores = np.array(fitness_scores)
        # Replace negative fitness scores with the minimum fitness score
        min_fitness = np.min(fitness_scores[np.logical_not(np.isneginf(fitness_scores))])
        fitness_scores = np.where(np.isneginf(fitness_scores), min_fitness, fitness_scores)
        total_fitness = np.sum(fitness_scores)
        probabilities = fitness_scores / total_fitness
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        selected_individuals = np.array(population)[selected_indices]
        return selected_individuals.tolist()

    def evaluation_function(self, q_current, q_next):
        qd = np.abs(q_next - q_current) / self.dt
        qdd = (qd - self.velocities[tuple(q_current)]) / self.dt

        qd_sign = np.sign(qd)
        qdd_sign = np.sign(qdd)

        qd = np.minimum(np.abs(qd), PUMA_VELOCITY_LIMITS) * qd_sign
        qdd = np.minimum(np.abs(qdd), PUMA_ACCELERATION_LIMITS) * qdd_sign

        if np.any(np.greater_equal(
                np.abs(torque(self.robot, q_next, qd, qdd)),
                PUMA_TORQUE_LIMITS)):
            return np.inf

        if tuple(q_next) not in self.velocities:
            self.velocities[tuple(q_next)] = qd

        return .9 * distance_cost(self.robot, q_current, q_next) + .1 * torque_cost(self.robot, q_current, qd, qdd)

    def fitness_function(self, individual):
        total_cost = 0
        for i in range(len(individual) - 1):
            q_current = individual[i]
            q_next = individual[i + 1]
            cost = self.evaluation_function(q_current, q_next)
            total_cost += cost
        return 1 / total_cost
