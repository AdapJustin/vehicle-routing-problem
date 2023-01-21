class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        """
        cost_matrix - distance of 1 point to another
        rank - rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]
