import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1c_problem:
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState, f=0):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState
        gameState.action = None
        self.start = self.getStartState()
        gameState.f = f
        self.food = self.startingGameState.getFood()
        self.num_of_food = self.startingGameState.getNumFood()
        self.foods = set()

        for x in range(self.food.width):
            for y in range(self.food.height):
                if self.food[x][y]:
                    self.foods.add((x, y))

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
            successors.append((state.generatePacmanSuccessor(action), action))

        return successors