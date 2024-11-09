import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    class Node:
        def __init__(self, position, type="Default"):
            self.position = position
            self.type = type
            self.action = {"North": None, "South": None, "East": None, "West": None}

        def __str__(self):
            return str(self.position) +  ", " + self.type + ", " + str(self.action) + "\n"

        def __repr__(self):
            return str(self)
        
    class Maze:
        def __init__(self, walls, food, pacman_position):
            self.walls = walls
            self.food = food
            self.graph = {}
            self.pacman_position = pacman_position

            self.food_list = food.asList()

            self.create_maze()
        
        def connect_upwards(self, position_1, position_2):
            self.graph[position_1].action["North"] = position_2
            self.graph[position_2].action["South"] = position_1

        def connect_leftwards(self, position_1, position_2):
            self.graph[position_1].action["East"] = position_2
            self.graph[position_2].action["West"] = position_1

        def create_node(self, current_position):
            new_node = q1b_problem.Node(current_position)
            self.graph[current_position] = new_node

            x, y = current_position

            for i in range(x - 1, -1, -1):
                if self.walls[i][y]:
                    break
                
                if (i, y) in self.graph:
                    self.connect_leftwards((i, y), current_position)
                    break

            for i in range(y - 1, -1, -1):
                if self.walls[x][i]:
                    break
                
                if (x, i) in self.graph:
                    self.connect_upwards((x, i), current_position)
                    break

        
        def create_maze(self):
            for x in range(1, self.walls.width -1):
                for y in range(1, self.walls.height - 1):
                    if self.food[x][y]:
                        self.create_node((x, y))
                        self.graph[(x, y)].type = "Food"
                    
                    elif self.pacman_position == (x, y):
                        self.create_node((x, y))
                        self.graph[(x, y)].type = "Pacman"
                    
                    elif not self.walls[x][y]:
                        up = self.walls[x][y + 1]
                        down = self.walls[x][y - 1]
                        left = self.walls[x - 1][y]
                        right = self.walls[x + 1][y]

                        wall_count = int(up) + int(down) + int(left) + int(right)
                        up_left = self.walls[x - 1][y + 1]
                        up_right = self.walls[x + 1][y + 1]
                        down_left = self.walls[x - 1][y - 1]
                        down_right = self.walls[x + 1][y - 1]


                        if wall_count >= 3 or wall_count == 1 or (wall_count == 2 and not ((up and down) or (left and right))) or (wall_count == 0 and (up_left or up_right or down_left or down_right)):
                            self.create_node((x, y))

        def __str__(self):
            graph_str = [[str(int(j)) for j in i] for i in self.walls]
            for node in self.graph:
                x, y = node
                graph_str[x][y] = str("N")
            
            return "\n".join(["".join(i) for i in graph_str])

    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState
        self.maze = self.Maze(gameState.getWalls(), gameState.getFood(), gameState.getPacmanPosition())
        self.food = self.startingGameState.getFood()
        self.num_of_food = self.startingGameState.getNumFood()
        self.foods = set(self.food.asList())


    @log_function
    def getStartState(self):
        "*** YOUR CODE HERE ***"
        return self.startingGameState

    @log_function
    def isGoalState(self, state):
        "*** YOUR CODE HERE ***"
        x, y = state.getPacmanPosition()
        return (x, y) in self.foods

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        "*** YOUR CODE HERE ***"
        actions = state.getLegalPacmanActions()
        successors = []
        for action in actions:
            successors.append(state.generatePacmanSuccessor(action))
            successors[-1].action = action
            successors[-1].prevState = state

        return successors

