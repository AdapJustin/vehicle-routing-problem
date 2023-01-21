class DesPoint:
    def __init__(self, number, xcoord, ycoord, assignedDepot):
        self.number = number
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.assignedDepot = assignedDepot

    def setAssignedDepot(self, assignedDepot):
        self.assignedDepot = assignedDepot

    def getAssignedDepot(self):
        return self.assignedDepot
