# Solving Dynamic Multi Depot Vehicle Routing Problem using Modified Hybrid Ant Colony Optimization
[![Generic badge](https://img.shields.io/badge/build-passed-<COLOR>.svg)](https://shields.io/)      [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This project contains 4 module algorithms which solves the Dynamic Multi Depot Vehicle Routing Problem with the basis of the Ant Colony Algorithm combined with different Genetic algorithms and Local Interchange algorithms.

## Introduction
> a few points to note!
- Everyday, thousands of trucks and motorcycles run around the city for logistical purposes. They go to different customers and stops from a starting point.
- Vehicle Routing Problem is similar to the Travelling Salesman Problem.
- Multi Depot means multiple starting points.
- K-Means will be used to identify what depot should handle which customer
- Ant Colony Optimization  is population-based metaheuristic that can be used to find approximate solutions to difficult optimization problems. Responsible for generating routes.

##### Route Optimization
-  3-Opt Algorithm - local interchange algorithm
-  Crossover Operator - genetic algorithm

## File Descriptions
1. Depot.py
    >   Depot object that contains the depot number, x and y coordinates of the depot, route, and route cost
2. DesPoint.py
    >   Destination point or customer object that contains the customer number, x and y coordinates,  and their assigned depot
3. Graph.py
    >   Graph object that contains the cost matrix and pheromone counts for each ranks or points
4. main.py
    >   Handles the preprocessing of the data set including clustering, parameter initialization and dynamism
5. HACO.py
    >   Basis algorithm which consists of ACO, Mutation Operation, and 2-Opt local interchange
6. MHACO_1.py
    >   New algorithm which consists of ACO, Crossover operation, and 3-Opt local interchange
7. MHACO_2.py
    >   Second Module of the new algorithm which consists of ACO, Crossover operation only
8. MHACO_3.py
    >   Third Module of the new algorithm which consists of ACO, 3-opt local interchange only

## Installation & Requirements
The algorithm implementation is based on the Python 3.9 language, the IDE used is PyCharm Community Edition 2022, and the computer configuration when testing was done is an Intel Core i7-8700, 16 GB RAM, and NVIDIA GeForce GTX 1660 running Windows 10(x64)

##### Minimum System Requirements:
   - 64-bit versions of Microsoft Windows 10, 8, 7 (SP1)
   - 4 GB RAM minimum, 8 GB RAM recommended
   - 1.5 GB hard disk space + at least 1 GB for caches
   - 1024×768 minimum screen resolution
   - Python 2.7, or Python 3.5 or newer
   - PyCharm 64-bit
   
##### Project Setup Instructions:
1. Download MHACO.zip and extract the files
2. Open the project folder in PyCharm
3. Install the following libraries using pip or conda:
            - openpyxl
            - pandas
            - matplotlib
            - numpy
            - sklearn

## How-to-use
Open main.py and configure the parameters found in the main function (bottom line of the code)  and Init region of the code.
 ```sh
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
```
 When satisfied with the given parameters run main.py. Results should a graph with information about the problem set and cost and run time of each algorithm.
 


 
 



