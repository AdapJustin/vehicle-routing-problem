import copy
import math
import random
import Graph
import numpy as np
from matplotlib import pyplot as plt


def calc_distance(points_list):
    total_sum = 0
    for i in range(len(points_list) - 1):
        point_a = points_list[i]
        point_b = points_list[i + 1]

        d = math.sqrt(
            math.pow(point_b[1] - point_a[1], 2) + math.pow(point_b[2] - point_a[2], 2)
        )

        total_sum += d

    point_a = points_list[0]
    point_b = points_list[-1]
    d = math.sqrt(math.pow(point_b[1] - point_a[1], 2) + math.pow(point_b[2] - point_a[2], 2))

    total_sum += d

    return total_sum


def crossover(best_solution, coord, Limiter):
    # Prepare date for GA
    for k in range(len(coord)):
        if len(coord[k]) > 2: coord[k].pop(0)
        coord[k].insert(0, str(k))
    POPULATION_SIZE = 4000
    TOURNAMENT_SELECTION_SIZE = 4
    CROSSOVER_RATE = 0.9
    if Limiter:
        ITERATION = 1
    if not Limiter:
        ITERATION = 100
    GA_coord = coord

    # Use ACO best solution
    GA_coord = [GA_coord[i] for i in best_solution[:-1]]
    lenGA_coord = len(GA_coord)
    total_sum = 0.0

    for i in range(len(GA_coord) - 1):
        GA_coordA = GA_coord[i]
        GA_coordB = GA_coord[i + 1]

        d = math.sqrt(math.pow(GA_coordB[1] - GA_coordA[1], 2) + math.pow(GA_coordB[2] - GA_coordA[2], 2))
        total_sum += d
    GA_coordA = GA_coord[0]
    GA_coordB = GA_coord[-1]
    d = math.sqrt(math.pow(GA_coordB[1] - GA_coordA[1], 2) + math.pow(GA_coordB[2] - GA_coordA[2], 2))
    total_sum += d
    population = []
    for i in range(POPULATION_SIZE):
        c = GA_coord.copy()
        random.shuffle(c)
        distance = total_sum
        population.append([distance, c])

    iteration_ctr = 0
    ga_cost_temp = 0
    for i in range(ITERATION):
        new_population = []
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])

        for k in range(int((len(population) - 2) / 2)):
            # CROSSOVER
            random_number = random.random()
            if random_number < CROSSOVER_RATE:
                parent_chromosome1 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]

                parent_chromosome2 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]

                point = random.randint(0, lenGA_coord - 1)

                child_chromosome1 = parent_chromosome1[1][0:point]
                for j in parent_chromosome2[1]:
                    if not (j in child_chromosome1):
                        child_chromosome1.append(j)

                child_chromosome2 = parent_chromosome2[1][0:point]
                for j in parent_chromosome1[1]:
                    if not (j in child_chromosome2):
                        child_chromosome2.append(j)
            # If crossover not happen
            else:
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]

            new_population.append([calc_distance(child_chromosome1), child_chromosome1])
            new_population.append([calc_distance(child_chromosome2), child_chromosome2])
        population = new_population
        ga_cost = sorted(population)[0][0]
        # For crossover convergence
        if ga_cost == ga_cost_temp:
            iteration_ctr += 1
        else:
            iteration_ctr = 0
        if iteration_ctr == 10:
            break
        ga_cost_temp = copy.deepcopy(ga_cost)
        #print('GA Iteration ' + str(i + 1) + ' -> ' + str(ga_cost))
    best_solution_ga = []
    for i in range(len(population[0][1])):
        best_solution_ga.append(int(population[0][1][i][0]))
    best_solution_ga.append(int(population[0][1][0][0]))
    for k in range(len(coord)):
        coord[k].pop(0)
    return ga_cost, best_solution_ga


class MHACO_2:
    """
    m - # of ants
    q -  pheromone intensity
    alpha - pheromone weight
    beta - Visibility weight
    rho - exploration rate
    generations - # of iterations
    strategy - pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
    """

    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int, strategy: int):
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, new_graph: Graph, new_coord, graph: Graph, coord, new_points):
        best_cost = float(9999)
        best_solution = []
        ant_cost = float('inf')
        for gen in range(self.generations):
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            # initial values
            former_best_cost = copy.copy(ant_cost)
            for ant in ants:
                for i in range(graph.rank):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < ant_cost:
                    ant_cost = ant.total_cost
                    if best_cost > ant_cost:
                        best_solution = [] + ant.tabu
                # update pheromone
                ant._update_pheromone_delta()
            if ant_cost == former_best_cost:
                ga_cost, best_solution_ga = crossover(best_solution, copy.deepcopy(coord), False)
                break
            else:
                ga_cost, best_solution_ga = crossover(best_solution, copy.deepcopy(coord), True)
            self._update_pheromone(graph, ants)
            best_solution = best_solution_ga
        #region dynamism
        if len(new_coord) > len(coord):
            # Fix solution for dynamism
            best_solution.pop(len(best_solution) - 1)
            for i in range(len(best_solution)):
                if best_solution[0] == len(best_solution) - 1:
                    break
                else:
                    best_solution.append(best_solution[0])
                    best_solution.pop(0)
            best_solution.append(best_solution[0])
            best_solution[0] = int(new_points[-1])
            best_solution[-1] = int(new_points[-1])
            for index, i in enumerate(new_points):
                if i == new_points[-1]:
                    break
                else:
                    best_solution.insert(len(best_solution) - 1, int(new_points[index]))
            ga_cost, best_solution_ga = crossover(best_solution, new_coord, False)
        # endregion
        print("\n============MHACO_2==============")
        print("ACO Generation: " + str(gen + 1))
        print('ACO cost: ' + str(ant_cost) + '\nACO solution: ' + str(best_solution))
        print('Crossover cost: ' + str(ga_cost) + '\nGA solution: ' + str(best_solution_ga))

        return best_solution_ga, ga_cost, new_coord


class _Ant(object):
    def __init__(self, mhaco: MHACO_2, graph: Graph):
        self.colony = mhaco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list for counting cost
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in range(graph.rank)]  # heuristic information
        start = graph.rank - 1  # change to make it start at depot ,add -1 if index out of bounds
        self.end = graph.rank - 1
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        # print(self.allowed)
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                i] ** self.colony.beta

        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                                   self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing

        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break

        if len(self.allowed) == 1:

            self.allowed.append(self.end)
            self.allowed.remove(selected)
            self.tabu.append(selected)
            self.total_cost += self.graph.matrix[self.current][selected]
            self.current = selected

        else:
            self.allowed.remove(selected)
            self.tabu.append(selected)
            self.total_cost += self.graph.matrix[self.current][selected]  # to get cost from point to point
            self.current = selected

    def checkProbability(self, randProbabilities, depot, allowed):
        selected = 1
        rand = random.random()
        for i, probability in enumerate(randProbabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        return selected

    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
