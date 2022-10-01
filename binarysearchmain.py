import numpy as np
from aipython import searchProblem, searchGeneric, searchBranchAndBound

# use searchProblem.py, searchGeneric.py, searchBranchAndBound.py
# searchProblem is where you input the problem and define start, goal
# searchGeneric does depth-first search
# searchBranchAndBound does a Branch and Bound search
import time
start_time = time.time()
class Neighbours():
    def __init__(self, arcs, goals, newnodes):
        self.arcs = arcs
        self.goals = goals
        self.newnodes = newnodes
    def findNeighbours(self,nodes):
        s = searchProblem
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
                                            to_node=(i,j+1),
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
                                            to_node=(i, j-1),
                                            cost=1
                                            ))
                # Goal cell
                if col[0] == '1':
                    goal = (i,j)
                    self.goals.add(goal)
        return self.arcs, self.newnodes, self.goals

def myheuristic(nodes, goals):
    distanceCost = 1
    returnNodes = {}
    for node in nodes:
        for goal in goals:
            dx = abs(node[0]-goal[0])
            dy = abs(node[1]-goal[1])
            # x/y value of a node minus the x/y value of a goal multiplied by cost
            # sum of absolute differences
            returnNodes[node] = distanceCost+(dx+dy)
    return returnNodes


def startingLocations(locations):
    locations = locations[3:]
    it = iter(locations)
    loc = [*zip(it, it)]
    # picking a random node to be my starting location
    return loc[5]

def generic(nodes, arcs, goals, start):
    # Depth First Search
    problem1 = searchProblem.Search_problem_from_explicit_graph(
        nodes=nodes,
        arcs= arcs,
        start=start,
        goals=goals
        ,hmap=myheuristic(nodes=nodes, goals=goals)
    )
    dfsSearch = searchGeneric.Searcher(problem=problem1)
    dfsSearch.search()
    # A*
    aStarSearch = searchGeneric.AStarSearcher(problem=problem1)
    #aStarSearch.search()

# Create a matrix to use as a new maze, and turn each value into binary length 5
def mazeMatrix(mazefile):
    mazefile = mazefile[3:]
    getbinary = lambda x, n: format(x, 'b').zfill(n)
    maze = []
    for i in range(15):
        row=[]
        for j in range(15):
            row.append(getbinary(mazefile[i * 15 + j],5))
        maze.append(row)

    return maze

for i in range(1,11):
    # Obtain mazes and process them.
    scmp = open("SCMP/SCMP1/mazes/M0_"+str(i)+".mz")
    scmp = list([int(x) for x in scmp.read().split(' ')])
    scmploc = open('SCMP/SCMP1/starting_locations.loc')
    scmploc = list([int(x) for x in scmploc.read().split(' ')])

    # Construct the neighbours class and build a matrix
    n = Neighbours(arcs = [], goals = set(), newnodes = set())
    matrix = mazeMatrix(scmp)
    # pick a starting location
    starts = startingLocations(scmploc)
    # Find all the neighbours using the matrix
    n.findNeighbours(matrix)
    # Searches go after matrix creation.
    generic(nodes=n.newnodes, arcs=n.arcs, goals=n.goals, start=starts)
print("--- %s seconds to run search ---" % (time.time() - start_time))

