#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#


def q1c_solver(problem: q1c_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result


class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.priority_queue = util.PriorityQueue()
        self.visited = {}
        self.discovered = {}
        self.startingProblem = None
        self.previous = {}
        self.path = []
        AStarData.foods = None
    
    def reset(self, current_state):
        self.path = self.path + backtrack(current_state, self.previous)
        self.priority_queue = util.PriorityQueue()
        self.priority_queue.push(current_state, 0)
        self.visited = {}
        self.discovered = {current_state.getPacmanPosition(): 0}
        self.previous = {current_state.getPacmanPosition(): (None, None)}

def astar_initialise(problem: q1c_problem):
    # YOUR CODE HERE
    astarData = AStarData()
    astarData.startingProblem = problem
    astarData.priority_queue.push(problem.getStartState(), 0)
    astarData.discovered = {problem.getStartState().getPacmanPosition(): 0}
    astarData.previous = {problem.getStartState().getPacmanPosition(): (None, None)}
    AStarData.foods = problem.foods
    return astarData

def astar_loop_body(problem: q1c_problem, astarData: AStarData):
    # YOUR CODE HERE
    if len(astarData.priority_queue.heap) == 0:
        return True, astarData.path
    
    current_state = astarData.priority_queue.pop()
    if astarData.startingProblem.isGoalState(current_state):
        AStarData.foods.remove(current_state.getPacmanPosition())
        astarData.reset(current_state)
        astarData.startingProblem.num_of_food -= 1
        
        if astarData.startingProblem.num_of_food == 0:
            return True, astarData.path
    
    astarData.visited[current_state.getPacmanPosition()] = 0

    for successor, action in astarData.startingProblem.getSuccessors(current_state):
        g = astarData.visited[current_state.getPacmanPosition()] + 1
        h = astar_heuristic(successor.getPacmanPosition(), AStarData.foods)
        f = g + h
        if successor.getPacmanPosition() not in astarData.discovered or g < astarData.discovered[successor.getPacmanPosition()]:
            astarData.discovered[successor.getPacmanPosition()] = g
            astarData.priority_queue.push(successor, f)
            astarData.previous[successor.getPacmanPosition()] = (current_state, action)

    return False, None
    

def astar_heuristic(current, goals):
    # YOUR CODE HERE
    return min([util.manhattanDistance(current, goal) for goal in goals])


def backtrack(state, previous):
    path = []
    while previous[state.getPacmanPosition()] != (None, None):
        state, action = previous[state.getPacmanPosition()]
        path.append(action)


    return path[::-1]
