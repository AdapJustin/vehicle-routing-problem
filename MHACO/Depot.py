class Depot:
    def __init__(self, number, xcoord, ycoord, route, routeCost):
        self.number = number
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.route = route
        self.routeCost = routeCost

    def setRoute(self, route):
        self.route = route

    def getRoute(self):
        return self.route

    def setRouteCost(self, cost):
        self.routeCost = cost

    def getRouteCost(self):
        return self.routeCost

