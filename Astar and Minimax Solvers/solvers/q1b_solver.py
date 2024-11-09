#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
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

from itertools import combinations
from game import Directions, Actions

class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.priority_queue = util.PriorityQueue()
        self.visited = {}
        self.startingProblem = None
        self.path = []
        AStarData.foods = None
        AStarData.heuristic_matrix = None
    
    def reset(self, current_node):
        self.path.extend(backtrack(current_node.position, AStarData.previous))
        self.priority_queue = util.PriorityQueue()
        self.priority_queue.push(current_node, 0)
        self.visited = {}
        AStarData.fCosts = {current_node.position: 0}
        AStarData.previous = {current_node.position: (None, None)}
        AStarData.heuristic_matrix = bfs_heuristic_generator(self.startingProblem)


def astar_initialise(problem: q1b_problem):
    # YOUR CODE HERE
    astarData = AStarData()
    astarData.startingProblem = problem
    starting_node = problem.maze.graph[problem.maze.pacman_position]
    AStarData.fCosts = {starting_node.position: 0}
    AStarData.previous = {starting_node.position: (None, None)}
    AStarData.foods = problem.foods
    AStarData.heuristic_matrix = bfs_heuristic_generator(problem)
    astarData.priority_queue.push(starting_node, 0)
    return astarData

def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    # YOUR CODE HERE
    
    if len(astarData.priority_queue.heap) == 0:
        return True, astarData.path
    
    current_node = astarData.priority_queue.pop()
    if current_node.position in astarData.visited:
        return False, []
    
    astarData.visited[current_node.position] = AStarData.fCosts[current_node.position] - astar_heuristic(current_node)
    if current_node.position in AStarData.foods:
        AStarData.foods.remove(current_node.position)
        problem.num_of_food -= 1

        astarData.reset(current_node)
        
        if astarData.startingProblem.num_of_food == 0:
            return True, astarData.path
        else:
            return False, []

    for action in current_node.action:
        if current_node.action[action] is None:
            continue
        
        next_node = problem.maze.graph[current_node.action[action]]
        h = astar_heuristic(next_node)
        g = astarData.visited[current_node.position] + util.manhattanDistance(current_node.position, next_node.position)
        f = g + h
        if next_node.position not in AStarData.fCosts or f < AStarData.fCosts[next_node.position]:
            AStarData.fCosts[next_node.position] = f
            AStarData.previous[next_node.position] = (current_node.position, action)
            astarData.priority_queue.push(next_node, f)

    return False, []

def astar_heuristic(current):
    # YOUR CODE HERE
    x, y = current.position
    return AStarData.heuristic_matrix[x][y]

def backtrack(current_position, previous):
    path = []
    while previous[current_position][0] is not None:
        previous_position, action = previous[current_position]
        for i in range(max(abs(current_position[0] - previous_position[0]), abs(current_position[1] - previous_position[1]))):
            path.append(action)
        
        current_position, action = previous[current_position]
    
    return path[::-1]



def bfs_heuristic_generator(problem:q1b_problem):
    """
    BFS heuristic generator, will generate a matrix that will give the shortest path from any food to any other point.
    """
    walls = problem.startingGameState.getWalls()
    food_grid = problem.food
    heuristic_matrix = [[float('inf') for _ in range(food_grid.height)] for _ in range(food_grid.width)]
    food_list = problem.foods
    
    """
    Go through every single food and bfs the whole matrix, keeping track of the minimum distance for each point
    """
    
    for food in food_list:
        queue = util.Queue()
        queue.push((food, 0))
        visited = set()
        while not queue.isEmpty():
            current, distance = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            if distance >= heuristic_matrix[current[0]][current[1]]:
                continue
            heuristic_matrix[current[0]][current[1]] = min(heuristic_matrix[current[0]][current[1]], distance)
            for action in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                next_position = (current[0] + action[0], current[1] + action[1])
                if next_position not in visited and not walls[next_position[0]][next_position[1]]:
                    queue.push((next_position, distance + 1))
    

    return heuristic_matrix
    