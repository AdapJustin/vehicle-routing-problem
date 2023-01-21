import copy
import math
import random
import Graph
import numpy as np
from matplotlib import pyplot as plt


# Function: Tour Distance
def calc_distance_opt(Xdata, point_tour):
    distance = 0
    for k in range(0, len(point_tour[0]) - 1):
        m = k + 1
        distance = distance + Xdata[point_tour[0][k] - 1, point_tour[0][m] - 1]
    return distance


def three_opt(cost_matrix, path, best_cost, Limiter):
    cost_matrix_opt = copy.deepcopy(cost_matrix)
    cost_matrix_opt = np.array(cost_matrix_opt)
    path_tour = [path, 9999]
    recursive_seeding = -1
    if recursive_seeding < 0:
        count = recursive_seeding - 1
    else:
        count = 0
    path_list = copy.deepcopy(path_tour)
    path_list_old = path_list[1] * 2
    iteration = 0
    ctr = 0
    while count < recursive_seeding:
        best_route = copy.deepcopy(path_list)
        best_route_2 = [[], 1]
        best_route_3 = [[], 1]
        best_route_4 = [[], 1]
        best_route_5 = [[], 1]
        seed = copy.deepcopy(path_list)
        for i in range(0, len(path_list[0]) - 3):
            for j in range(i + 1, len(path_list[0]) - 2):
                for k in range(j + 1, len(path_list[0]) - 1):
                    best_route_2[0] = best_route[0][:i + 1] + best_route[0][j + 1:k + 1] + best_route[0][i + 1:j + 1] + \
                                      best_route[0][k + 1:]
                    best_route_2[1] = calc_distance_opt(cost_matrix_opt, best_route_2)
                    best_route_3[0] = best_route[0][:i + 1] + list(reversed(best_route[0][i + 1:j + 1])) + list(
                        reversed(best_route[0][j + 1:k + 1])) + best_route[0][k + 1:]
                    best_route_3[1] = calc_distance_opt(cost_matrix_opt, best_route_3)
                    best_route_4[0] = best_route[0][:i + 1] + list(reversed(best_route[0][j + 1:k + 1])) + best_route[
                        0][i + 1:j + 1] + best_route[0][k + 1:]
                    best_route_4[1] = calc_distance_opt(cost_matrix_opt, best_route_4)
                    best_route_5[0] = best_route[0][:i + 1] + best_route[0][j + 1:k + 1] + list(
                        reversed(best_route[0][i + 1:j + 1])) + best_route[0][k + 1:]
                    best_route_5[1] = calc_distance_opt(cost_matrix_opt, best_route_5)

                    if best_route_2[1] < best_route[1]:
                        best_route = copy.deepcopy(best_route_2)
                    elif best_route_3[1] < best_route[1]:
                        best_route = copy.deepcopy(best_route_3)
                    elif best_route_4[1] < best_route[1]:
                        best_route = copy.deepcopy(best_route_4)
                    elif best_route_5[1] < best_route[1]:
                        best_route = copy.deepcopy(best_route_5)
                if best_route[1] < path_list[1]:
                    path_list = copy.deepcopy(best_route)
                best_route = copy.deepcopy(seed)
        count = count + 1
        iteration = iteration + 1
        #print('3-opt Iteration ' + str(iteration) + ' -> ' + str(path_list[1]))
        if path_list_old > path_list[1] and recursive_seeding < 0:
            path_list_old = path_list[1]
            count = -2
            recursive_seeding = -1
        elif path_list[1] >= path_list_old and recursive_seeding < 0:
            count = -1
            recursive_seeding = -2
        ctr += 1
        if Limiter:
            if path_list[1] < best_cost:
                break
    best_solution_opt = path_list[0]
    best_cost_opt = path_list[1]
    # Fix solution ordering
    for i in range(len(best_solution_opt)):
        if best_solution_opt[i] == 0:
            best_solution_opt[i] = len(best_solution_opt) - 2
        else:
            best_solution_opt[i] -= 1
    return best_cost_opt, best_solution_opt


class MHACO_3:
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
                opt_cost, best_solution_opt = three_opt(graph.matrix, best_solution, ant_cost, False)
                break
            else:
                opt_cost, best_solution_opt = three_opt(graph.matrix, best_solution, ant_cost, True)
            self._update_pheromone(graph, ants)
            best_solution = best_solution_opt
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
            opt_cost, best_solution_opt = three_opt(new_graph.matrix, best_solution, best_cost, False)
        # endregion

        print("\n============MHACO_3==============")
        print("ACO Generation: " + str(gen + 1))
        print('ACO cost: ' + str(ant_cost) + '\nACO solution: ' + str(best_solution))
        print('3-opt cost: ' + str(opt_cost) + '\n3-opt solution: ' + str(best_solution_opt))

        return best_solution_opt, opt_cost, new_coord


class _Ant(object):
    def __init__(self, mhaco: MHACO_3, graph: Graph):
        self.colony = mhaco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list for counting cost
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
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
