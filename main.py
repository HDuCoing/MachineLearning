import numpy as np
from aipython import searchProblem, searchGeneric, searchBranchAndBound

# use searchProblem.py, searchGeneric.py, searchBranchAndBound.py
# searchProblem is where you input the problem and define start, goal
# searchGeneric does depth-first search
# searchBranchAndBound does a Branch and Bound search
scmp = open("SCMP/SCMP4/mazes/M100_1.mz")
scmp = list([int(x) for x in scmp.read().split(' ')])
scmploc = open('SCMP/SCMP4/starting_locations.loc')
scmploc = list([int(x) for x in scmploc.read().split(' ')])

class Neighbours():
    def __init__(self, arcs, goals, newnodes):
        self.arcs = arcs
        self.goals = goals
        self.newnodes = newnodes
    def findNeighbours(self,nodes):
        s = searchProblem
        '''
        for every row, col in maze:
            take binary value and data value
            check all 4 possible directions to find valid neighbour node
            for each for the 4 neighbouring nodes within border
            if no walls between them make an arc
        :return:
        '''
        # check all 4 directions to find valid neighbour node
        for i, row in enumerate(nodes):
            for j, col in enumerate(row):
                self.newnodes.add((i,j))
                # Wall on upper side
                if col[4] != '1':
                    self.arcs.append(s.Arc(from_node=(i, j),
                                            to_node=(i-1, j),
                                            cost=1
                                            ))
                # Wall on right side
                if col[3] != '1':
                    self.arcs.append(s.Arc(from_node=(i, j),
                                            to_node=(i,j-1),
                                            cost=1
                                            ))
                # Wall on lower side
                if col[2] != '1':
                    self.arcs.append(s.Arc(from_node=(i,j),
                                            to_node=(i+1, j),
                                            cost=1
                                            ))
                # Wall on left side
                if col[1] != '1':
                    self.arcs.append(s.Arc(from_node=(i,j),
                                            to_node=(i, j+1),
                                            cost=1
                                            ))
                # Goal cell
                if col[0] == '1':
                    goal = (i,j)
                    self.goals.add(goal)
        return self.arcs, self.newnodes, self.goals


def startingLocations(locations):
    locations = locations[3:]
    it = iter(locations)
    loc = [*zip(it, it)]
    # picking a random node to be my goal
    return loc[4]

def generic(nodes, arcs, goals, start):
    # Depth First Search
    problem1 = searchProblem.Search_problem_from_explicit_graph(
        nodes=nodes,
        arcs= arcs,
        start=start,
        goals=goals
    )
    dfsSearch = searchGeneric.Searcher(problem=problem1)
    dfsSearch.search()
    # A*
    aStarSearch = searchGeneric.AStarSearcher(problem=problem1)
    aStarSearch.search()

# Create a matrix to use as a new maze, and turn each value into binary length 5
def mazeMatrix(mazefile):
    getbinary = lambda x, n: format(x, 'b').zfill(n)
    maze = []
    nrow = mazefile[0]
    ncol = mazefile[1]
    for i in range(nrow):
        row=[]
        for j in range(ncol):
            row.append(getbinary(mazefile[i * nrow + j],5))
        maze.append(row)
    return maze

# Construct the neighbours class and build a matrix
n = Neighbours(arcs = [], goals = set(), newnodes = set())
matrix = mazeMatrix(scmp)
# pick a starting location
starts = startingLocations(scmploc)
# Find all the neighbours using the matrix
n.findNeighbours(matrix)
# Searches go after matrix creation.
generic(nodes=n.newnodes, arcs=n.arcs, goals=n.goals, start=starts)