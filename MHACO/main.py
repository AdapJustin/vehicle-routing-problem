import copy
import os

os.environ["OMP_NUM_THREADS"] = '1'
import time
import Depot
import DesPoint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import Graph
import random

from matplotlib.offsetbox import AnchoredText
from sklearn.cluster import KMeans
from MHACO_1 import MHACO_1
from MHACO_2 import MHACO_2
from MHACO_3 import MHACO_3
from HACO import HACO


def eucDistance(point1X, point2X, point1Y, point2Y):
    return math.sqrt((point1X - point2X) ** 2 + (point1Y - point2Y) ** 2)


def euclidean_distance(x, y):
    distance = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    return distance


def plot(pointsForRouting, route, depotColor, subplot):
    substring = "Depot "
    xplot = []
    yplot = []

    for index, i in enumerate(route):
        xplot.append(int(pointsForRouting[i][0]))
        yplot.append(int(pointsForRouting[i][1]))
        subplot.scatter(int(pointsForRouting[i][0]), int(pointsForRouting[i][1]), marker='.', s=100,color=depotColor)

    u = np.diff(xplot)
    v = np.diff(yplot)
    pos_x = xplot[:-1] + u
    pos_y = yplot[:-1] + v
    norm = np.sqrt(u ** 2 + v ** 2)
    subplot.plot(xplot, yplot, color=depotColor)
    subplot.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=1, pivot="tip", color=depotColor,
                   scale_units='inches', scale=5.0)


def computeCostMatrix(depot: Depot, points_cm_list: list, depot_index):
    points_for_cost_matrix = []
    points_ref_for_routing = []
    cost_matrix_list = []

    for k in points_cm_list:
        if k.assignedDepot == (depot_index + 1):
            points_for_cost_matrix.append([k.xcoord, k.ycoord])
            points_ref_for_routing.append(k)
    points_for_cost_matrix.append([depot.xcoord, depot.ycoord])  # last index is always for depot [point1, point2, point3, ...., pointn, depot]
    points_ref_for_routing.append(depot)
    rank_cm = len(points_for_cost_matrix)
    for k in range(rank_cm):
        row = []
        for j in range(rank_cm):
            row.append(
                eucDistance(points_for_cost_matrix[k][0], points_for_cost_matrix[j][0], points_for_cost_matrix[k][1],points_for_cost_matrix[j][1]))
        cost_matrix_list.append(row)

    return cost_matrix_list, rank_cm, points_for_cost_matrix, points_ref_for_routing


def recomputeCostMatrix(point_coord):
    points_for_cost_matrix = []
    cost_matrix_list = []
    for k in point_coord:
        points_for_cost_matrix.append([k[0], k[1]])

    rank_cm = len(points_for_cost_matrix)
    for k in range(rank_cm):
        row = []
        for j in range(rank_cm):
            row.append(euclidean_distance(points_for_cost_matrix[k], points_for_cost_matrix[j]))
        cost_matrix_list.append(row)

    return cost_matrix_list, rank_cm, points_for_cost_matrix


def pathConversion(route, points_ref_for_Routing, point_coords, subplot_algo):
    # path to point conversions
    actualRoute = []
    # comment for dynamism
    for convertedPoint in route:
        if convertedPoint == max(route):
            depotNum = "Depot " + str(int(points_ref_for_Routing[-1].number))
            actualRoute.append(depotNum)
        elif convertedPoint >= len(points_ref_for_Routing) - 1:
            subplot_algo.annotate("DP " + str(convertedPoint), (point_coords[convertedPoint][0], point_coords[convertedPoint][1]))
            actualRoute.append(convertedPoint)

        else:
            actualRoute.append(points_ref_for_Routing[convertedPoint].number)

    # Fix path ordering of GA solution
    if not isinstance(actualRoute[0], str):
        actualRoute.pop(len(actualRoute) - 1)
        while not isinstance(actualRoute[0], str):
            if not isinstance(actualRoute[0], str):
                actualRoute.append(actualRoute[0])
                actualRoute.pop(0)
            if isinstance(actualRoute[0], str):
                actualRoute.append(actualRoute[0])
                break
    return actualRoute


def dynamism(set1_points, depots_centroid, degrees_of_dynamism):
    init_points = len(set1_points)
    if degrees_of_dynamism == 0:
        km = KMeans(n_clusters=len(depots_centroid), init=np.array(depots_centroid), max_iter=1, n_init=1)
        label = km.fit_predict(set1_points)
        return [], label


    all_x = []
    all_y = []

    for point in set1_points:
        all_x.append(point[0])
        all_y.append(point[1])

    max_x = max(all_x)
    max_y = max(all_y)
    min_x = min(all_x)
    min_y = min(all_y)

    # generate the points
    num_points = (len(set1_points) * degrees_of_dynamism) / len(depots_centroid)
    new_points = []
    kmeans_points = []

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    print("total number of dynamic points = ", int(num_points))
    for i in range(int(num_points)):
        new_points.append(DesPoint.DesPoint(i + 1, random.randint(min_x, max_x), random.randint(min_y, max_y), None))

    for x in new_points:
        kmeans_points.append([x.xcoord, x.ycoord])
        set1_points.append([x.xcoord, x.ycoord])

    km = KMeans(n_clusters=len(depots_centroid), init=np.array(depots_centroid), max_iter=1, n_init=1)
    label = km.fit_predict(set1_points)

    for index, i in enumerate(new_points):
        despoint2 = i
        despoint2.setAssignedDepot(label[init_points + (index)] + 1)
        new_points.pop(index)
        new_points.insert(index, despoint2)

    return new_points, label


def dmdvrp(data_set, dod, aco_param):
    start_cluster_time = time.perf_counter()
    print("\n----------------------------------------------------------------")
    print("Problem Set ", data_set+1)
    print("Degrees of Dynamism", dod)
    figure, subplot_algo = plt.subplots(2, 2, figsize=(6, 6), dpi = 100)
    figure.tight_layout()
    # subplot_algo = ax
    # region GetData
    problemSet = pd.read_excel('Data Sets/Data_Set-MHACO.xlsx', sheet_name=data_set, header=[0])
    set1 = []
    set1Depots = []
    xCoord = "x coordinate"
    yCoord = "y coordinate"
    custNum = "Customer Number"
    depotNum = "Number of Depots"
    depotXCoord = "Depot x coordinate"
    depotYCoord = "Depot y coordinate"
    for x in range(problemSet[custNum].size):
        set1.append(DesPoint.DesPoint(problemSet[custNum].iloc[x], problemSet[xCoord].iloc[x], problemSet[yCoord].iloc[x], None))
    for x in range(problemSet[depotNum].size):
        if pd.isna(problemSet[depotNum].iloc[x]):
            break
        set1Depots.append(Depot.Depot(problemSet[depotNum].iloc[x], problemSet[depotXCoord].iloc[x], problemSet[depotYCoord].iloc[x], None, None))
    # endregion
    # region Cluster
    #
    # subplot_algo.figure(figsize=(10, 10), dpi=100)

    depotsCentroid = []
    points = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    for x in set1Depots:
        depotsCentroid.append([x.xcoord, x.ycoord])

    for x in set1:
        points.append([x.xcoord, x.ycoord])

    #region Dynamism
    dynamic_points, label = dynamism(points, depotsCentroid, dod)

    #endregion
    u_labels = np.unique(label)
    for index, i in enumerate(set1):
        despoint2 = i
        despoint2.setAssignedDepot(label[index] + 1)
        set1.pop(index)
        set1.insert(index, despoint2)

    for despoints in set1:
        subplot_algo[0, 0].annotate(despoints.number, (despoints.xcoord, despoints.ycoord))
        subplot_algo[0, 1].annotate(despoints.number, (despoints.xcoord, despoints.ycoord))
        subplot_algo[1, 0].annotate(despoints.number, (despoints.xcoord, despoints.ycoord))
        subplot_algo[1, 1].annotate(despoints.number, (despoints.xcoord, despoints.ycoord))

    for depots in set1Depots:
        subplot_algo[0, 0].scatter(depots.xcoord, depots.ycoord, color=colors[int(depots.number - 1)], marker='s', s=200)
        subplot_algo[0, 0].annotate(int(depots.number), (depots.xcoord, depots.ycoord))
        subplot_algo[0, 1].scatter(depots.xcoord, depots.ycoord, color=colors[int(depots.number - 1)], marker='s',s=200)
        subplot_algo[0, 1].annotate(int(depots.number), (depots.xcoord, depots.ycoord))
        subplot_algo[1, 0].scatter(depots.xcoord, depots.ycoord, color=colors[int(depots.number - 1)], marker='s', s=200)
        subplot_algo[1, 0].annotate(int(depots.number), (depots.xcoord, depots.ycoord))
        subplot_algo[1, 1].scatter(depots.xcoord, depots.ycoord, color=colors[int(depots.number - 1)], marker='s',s=200)
        subplot_algo[1, 1].annotate(int(depots.number), (depots.xcoord, depots.ycoord))
    # endregion

    cluster_time = time.perf_counter() - start_cluster_time

    # region solve
    overall_best_cost_mhaco_1 = 0
    overall_best_cost_mhaco_2 = 0
    overall_best_cost_mhaco_3 = 0
    overall_best_cost_haco = 0

    overall_runtime_mhaco_1 = 0
    overall_runtime_mhaco_2 = 0
    overall_runtime_mhaco_3 = 0
    overall_runtime_haco = 0

    for index, n in enumerate(set1Depots):
            start_init_time = time.perf_counter()
            print("----------------------------------------------------------------")
            print("Cluster ", index + 1)
            print("Color ", colors[int(n.number - 1)])
            print("----------------------------------------------------------------")
            cost_matrix, rank, pointsForCostMatrix, pointsRefForRouting = computeCostMatrix(n, set1, index)
            pointsForCostMatrixCopy = copy.deepcopy(pointsForCostMatrix)
            graph = Graph.Graph(cost_matrix, rank)

            cluster_dynamic_points = []
            for x in dynamic_points:
                if x.assignedDepot == n.number:
                    cluster_dynamic_points.append(x)
            init_time = time.perf_counter() - start_init_time

            new_matrix = []
            new_rank = []
            new_points_for_cost_matrix = []
            new_graph = copy.copy(graph)
            algos = ["MHACO_1", "MHACO_2", "MHACO_3", "HACO"]
            for algo in algos:
                if algo == "MHACO_1":
                    # MHACO_1
                    start_mhaco_1 = time.perf_counter()

                    mhaco_1 = MHACO_1(aco_param[0], aco_param[1], aco_param[2], aco_param[3], aco_param[4], aco_param[5], aco_param[6])

                    MHACO_1_route, MHACO_1_cost, point_coords, new_points = mhaco_1.solve(graph, pointsForCostMatrixCopy, cluster_dynamic_points)

                    plot(point_coords, MHACO_1_route, colors[int(n.number - 1)], subplot_algo[0, 0])
                    actualRoute_mhaco_1 = pathConversion(MHACO_1_route, pointsRefForRouting, point_coords, subplot_algo[0, 0])

                    overall_best_cost_mhaco_1 += MHACO_1_cost
                    mhaco_1_time = time.perf_counter() - start_mhaco_1
                    overall_runtime_mhaco_1 += mhaco_1_time + init_time

                    new_matrix, new_rank, new_points_for_cost_matrix = recomputeCostMatrix(point_coords)
                    new_graph = Graph.Graph(new_matrix, new_rank)
                elif algo == "MHACO_2":
                    # MHACO_2
                    start_mhaco_2 = time.perf_counter()
                    mhaco_2 = MHACO_2(aco_param[0], aco_param[1], aco_param[2], aco_param[3], aco_param[4], aco_param[5], aco_param[6])
                    MHACO_2_route, MHACO_2_cost, MHACO_2_point_coords = mhaco_2.solve(new_graph, new_points_for_cost_matrix, graph, pointsForCostMatrixCopy, new_points)
                    plot(MHACO_2_point_coords, MHACO_2_route, colors[int(n.number - 1)], subplot_algo[0, 1])
                    actualRoute_mhaco_2 = pathConversion(MHACO_2_route, pointsRefForRouting, MHACO_2_point_coords, subplot_algo[0, 1])

                    overall_best_cost_mhaco_2 += MHACO_2_cost
                    mhaco_2_time = time.perf_counter() - start_mhaco_2
                    overall_runtime_mhaco_2 += mhaco_2_time + init_time
                elif algo == "MHACO_3":
                    # MHACO_3
                    start_mhaco_3 = time.perf_counter()
                    mhaco_3 = MHACO_3(aco_param[0], aco_param[1], aco_param[2], aco_param[3], aco_param[4], aco_param[5], aco_param[6])
                    MHACO_3_route, MHACO_3_cost, MHACO_3_point_coords = mhaco_3.solve(new_graph, new_points_for_cost_matrix, graph, pointsForCostMatrixCopy, new_points)
                    plot(MHACO_3_point_coords, MHACO_3_route, colors[int(n.number - 1)], subplot_algo[1, 0])
                    actualRoute_mhaco_3 = pathConversion(MHACO_3_route, pointsRefForRouting, MHACO_3_point_coords, subplot_algo[1, 0])

                    overall_best_cost_mhaco_3 += MHACO_3_cost
                    mhaco_3_time = time.perf_counter() - start_mhaco_3
                    overall_runtime_mhaco_3 += mhaco_3_time + init_time
                elif algo == "HACO":
                    # HACO
                    start_haco = time.perf_counter()

                    haco = HACO(aco_param[0], aco_param[1], aco_param[2], aco_param[3], aco_param[4], aco_param[5], aco_param[6])
                    HACO_route, HACO_cost, HACO_point_coords = haco.solve(new_graph, new_points_for_cost_matrix, graph, pointsForCostMatrixCopy, new_points)
                    plot(HACO_point_coords, HACO_route, colors[int(n.number - 1)], subplot_algo[1, 1])
                    actualRoute_haco = pathConversion(HACO_route, pointsRefForRouting, HACO_point_coords, subplot_algo[1, 1])

                    overall_best_cost_haco += HACO_cost
                    haco_time = time.perf_counter() - start_haco
                    overall_runtime_haco += haco_time + init_time

            # delegate routes and costs only uses mhaco_1
            n.setRoute(actualRoute_mhaco_1)
            n.setRouteCost(MHACO_1_cost)
            set1Depots.pop(index)
            set1Depots.insert(index, n)

            print("----------------------------------------------------------------")
            print("MHACO_1")
            print('Best Cost: {}'.format(MHACO_1_cost))
            print("Final Route: {}".format(MHACO_1_route))
            print("Converted Route: {}".format(actualRoute_mhaco_1))
            print("MHACO_2")
            print('Best Cost: {}'.format(MHACO_2_cost))
            print("Final Route: {}".format(MHACO_2_route))
            print("Converted Route: {}".format(actualRoute_mhaco_2))
            print("MHACO_3")
            print('Best Cost: {}'.format(MHACO_3_cost))
            print("Final Route: {}".format(MHACO_3_route))
            print("Converted Route: {}".format(actualRoute_mhaco_3))
            print("HACO")
            print('Best Cost: {}'.format(HACO_cost))
            print("Final Route: {}".format(HACO_route))
            print("Converted Route: {}".format(actualRoute_haco))
            print("----------------------------------------------------------------")

    # endregion

    print("MHACO_1 Overall Cost: {}".format(round(overall_best_cost_mhaco_1, 2)))
    print("Execution time: %s seconds" % round(overall_runtime_mhaco_1, 2))

    print("MHACO_2 Overall Cost: {}".format(round(overall_best_cost_mhaco_2, 2)))
    print("Execution time: %s seconds" % round(overall_runtime_mhaco_2, 2))

    print("MHACO_3 Overall Cost: {}".format(round(overall_best_cost_mhaco_3, 2)))
    print("Execution time: %s seconds" % round(overall_runtime_mhaco_3, 2))

    print("HACO Overall Cost: {}".format(round(overall_best_cost_haco, 2)))
    print("Execution time: %s seconds" % round(overall_runtime_haco, 2))
    print("----------------------------------------------------------------")

    return round(overall_best_cost_mhaco_1, 2), round(overall_runtime_mhaco_1, 2), round(overall_best_cost_mhaco_2, 2), round(
        overall_runtime_mhaco_2, 2), round(overall_best_cost_mhaco_3, 2), round(overall_runtime_mhaco_3, 2), round(overall_best_cost_haco, 2), round(
        overall_runtime_haco, 2), subplot_algo,figure, len(dynamic_points), len(set1), len(set1Depots)


def main():
    #region Initilize parameters
    #data range [0-18]
    data_set = 1
    #dod range [0-1]
    degrees_of_dynamism = 0

    """
    ACO parameters
    m - # of ants
    q -  pheromone intensity
    alpha - pheromone weight
    beta - Visibility weight
    rho - exploration rate
    generations - # of iterations
    strategy - pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
    """
    aco_param = [30, 100, 2.0, 1.0, 0.8, 10, 0]
    #endregion

    mhaco_1_cost, mhaco_1_runtime, mhaco_2_cost, mhaco_2_runtime, mhaco_3_cost, mhaco_3_runtime, haco_cost, haco_runtime, mainplt, fig, totalDP, initialPoints, numDepots = dmdvrp(data_set, degrees_of_dynamism,aco_param)

    mainplt[0, 0].set_title("MHACO_1 (aco, crossover, 3-opt)")
    mainplt[0, 1].set_title("MHACO_2 (aco, crossover)")
    mainplt[1, 0].set_title("MHACO_3 (aco, 3-opt)")
    mainplt[1, 1].set_title("HACO (aco, mutation, 2-opt)")
    text = """
            \n
            Problem Set: {} \n
            Degrees of Dynamism: {}\n
            Number of Depots: {}\n
            Initial Points: {}\n
            Number of Dynamic Points: {}\n
            Total Points: {}
            \n
            MHACO_1 cost: {}\n
            MHACO_1 runtime: {} seconds
            \n
            MHACO_2 cost: {} \n
            MHACO_2 runtime: {} seconds  
            \n         
            MHACO_3 cost: {}\n
            MHACO_3 runtime: {} seconds
            \n
            HACO cost: {}\n
            HACO runtime: {} seconds 
           """.format(data_set+1, degrees_of_dynamism,numDepots, initialPoints, totalDP, initialPoints + totalDP, mhaco_1_cost, mhaco_1_runtime, mhaco_2_cost, mhaco_2_runtime, mhaco_3_cost, mhaco_3_runtime,
                      haco_cost, haco_runtime)

    mainplt[0, 0].axis("off")
    mainplt[0, 1].axis("off")
    mainplt[1, 0].axis("off")
    mainplt[1, 1].axis("off")
    plt.subplots_adjust(left=0.3)
    plt.text(0.02, 0.4, text, fontsize=10, horizontalalignment='left', verticalalignment='bottom', transform=plt.gcf().transFigure)
    plt.show()



if __name__ == '__main__':
    main()
