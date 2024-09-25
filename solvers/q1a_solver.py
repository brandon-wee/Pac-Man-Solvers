#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.priority_queue = util.PriorityQueue()
        self.visited = {}
        self.discovered = {}
        self.previous = {}
    
def astar_initialise(problem: q1a_problem):
    # YOUR CODE HERE
    astarData = AStarData()
    astarData.priority_queue.push(problem.getStartState(), 0)
    astarData.discovered = {problem.getStartState(): 0}
    astarData.previous = {problem.getStartState(): (None, None)}
    return astarData

def astar_loop_body(problem: q1a_problem, astarData: AStarData):
    # YOUR CODE HERE
    if astarData.priority_queue.isEmpty():
        return True, []
    
    current_state = astarData.priority_queue.pop()
    cost = astarData.discovered[current_state]
    astarData.visited[current_state] = cost

    if problem.isGoalState(current_state):
        return True, backtrack(current_state, astarData.previous)
    
    for new_state, action in problem.getSuccessors(current_state):
        if new_state in astarData.visited:
            continue
        
        h = astar_heuristic(new_state, problem.goal)
        g = cost + 1
        f = g + h

        if new_state not in astarData.discovered or g < astarData.discovered[new_state]:
            astarData.priority_queue.push(new_state, f)
            astarData.discovered[new_state] = g
            astarData.previous[new_state] = (current_state, action)
        
    return False, []

def astar_heuristic(current, goal):
    # YOUR CODE HERE
    return util.manhattanDistance(current, goal)

def backtrack(state, previous):
    path = []
    while previous[state] != (None, None):
        state, action = previous[state]
        path.append(action)
    
    return path[::-1]