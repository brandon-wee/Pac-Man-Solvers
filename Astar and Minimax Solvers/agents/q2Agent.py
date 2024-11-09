import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()



class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = betterEvaluationFunction
        self.depth = int(depth)
        self.num_of_food = None
        Q2_Agent.food_matrix = None
        Q2_Agent.ghost_matrix = None
        Q2_Agent.closest_ghost = None
        Q2_Agent.score_not_increasing = 0
        Q2_Agent.current_score = 0

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        "*** YOUR CODE HERE ***"
        if self.num_of_food is None:
            self.num_of_food = gameState.getNumFood()
            Q2_Agent.food_matrix = food_heuristic_generator(gameState)
        
        if self.num_of_food != gameState.getNumFood():
            self.num_of_food = gameState.getNumFood()
            Q2_Agent.food_matrix = food_heuristic_generator(gameState)
        
        Q2_Agent.ghost_matrix = ghost_heuristic_generator(gameState)
        the_move = self.minimax(gameState)[1]
        Q2_Agent.current_score = gameState.getScore()
        next_score = gameState.generatePacmanSuccessor(the_move).getScore()
        if Q2_Agent.current_score <= next_score:
            Q2_Agent.score_not_increasing += 1
        else:
            Q2_Agent.score_not_increasing = 0
        
        print(the_move)
        return the_move
        # raise NotImplementedError("Ye")

    def minimax(self, gameState, alpha=-float('inf'), beta=float('inf'), agentIndex=0):
        if gameState.isWin() or gameState.isLose() or agentIndex // gameState.getNumAgents() == self.depth:
            return [betterEvaluationFunction(gameState), gameState]

        if agentIndex % gameState.getNumAgents() != 0:
            score = float('inf')
            best_move = None
            actions = gameState.getLegalActions(agentIndex % (gameState.getNumAgents()))

            for action in actions:
                successor = gameState.generateSuccessor(agentIndex % (gameState.getNumAgents()), action)
                new_score, _  = self.minimax(successor, alpha=alpha, beta=beta, agentIndex=agentIndex + 1)
                beta = min(beta, new_score)
                if new_score < score:
                    score = new_score
                    best_move = action
                if beta <= alpha:
                    break
            return [score, best_move]
        else:
            score = - float('inf')
            best_move = None
            actions = gameState.getLegalActions(agentIndex  % (gameState.getNumAgents()))
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex  % (gameState.getNumAgents()), action)
                new_score, _,  = self.minimax(successor, alpha=alpha, beta=beta, agentIndex=agentIndex + 1)

                alpha = max(alpha, new_score)
                if new_score > score:
                    score = new_score
                    best_move = action
                if alpha >= beta:
                    break
            
            return [score, best_move]


    
def betterEvaluationFunction(currentGameState):
    """
      An improved evaluation function to make Pacman less scared of ghosts,
      more focused on collecting dots while dodging ghosts effectively.
    """
    pacmanPos = currentGameState.getPacmanPosition()

    ghostStates = currentGameState.getGhostStates()


    # Base score is the current game score
    score = 10 * currentGameState.getScore()

    # Reward for moving towards food
    if Q2_Agent.food_matrix[pacmanPos[0]][pacmanPos[1]] != 0:
        score += 100 / Q2_Agent.food_matrix[pacmanPos[0]][pacmanPos[1]]
    else:
        score += 100
    
    # Ghost proximity and dodging strategy
    ghostPenalty = 0
    safeDistance = 5  # Define a safe distance threshold
    Q2_Agent.closest_ghost = float('inf')
    for i, ghostState in enumerate(ghostStates):
        scared_time = ghostState.scaredTimer
        distanceToGhost = Q2_Agent.ghost_matrix[pacmanPos[0]][pacmanPos[1]]
        if scared_time == 0 and distanceToGhost < Q2_Agent.closest_ghost:
            Q2_Agent.closest_ghost = distanceToGhost

        if distanceToGhost == 0:
            distanceToGhost = 1


        if scared_time > 0:
            # Encourage Pacman to eat ghosts when they are scared
            score += 200 / distanceToGhost
        else:
            if distanceToGhost < 2:
                ghostPenalty -= 50  # High penalty for being too close to a ghost
            elif distanceToGhost < safeDistance:
                ghostPenalty -= 20 / distanceToGhost  # Reduced penalty for being close but not too close
            else:
                ghostPenalty -= 2 / distanceToGhost  # Light penalty for being within safe distance

    # Encourage moving towards capsules
    capsuleList = currentGameState.getCapsules()
    if capsuleList:
        closestCapsuleDistance = min([manhattanDistance(pacmanPos, capsule) for capsule in capsuleList])
        score += 20 / closestCapsuleDistance

    # Penalize not progressing in the game unless close to a ghost
    if Q2_Agent.score_not_increasing > 25:
        if Q2_Agent.closest_ghost < safeDistance:
            score -= 10
        else:
            score -= 100


    # Combine all factors into the final score
    finalScore = score + ghostPenalty
 
    # Winning state bonus
    if currentGameState.isWin():
        finalScore += 1000000000000

    if currentGameState.isLose():
        finalScore -= 1000000000000

    return finalScore



def food_heuristic_generator(gameState: GameState):
    """
    BFS heuristic generator, will generate a matrix that will give the shortest path from any food to any other point.
    """
    walls = gameState.getWalls()
    food_grid = gameState.getFood()
    heuristic_matrix = [[float('inf') for _ in range(food_grid.height)] for _ in range(food_grid.width)]
    food_list = food_grid.asList()
    
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

def ghost_heuristic_generator(gameState):
    """
    Does the same thing as the food bfs heuristic generator, but for the ghost instead of the food.
    """
    walls = gameState.getWalls()
    ghost_states = gameState.getGhostStates()
    heuristic_matrix = [[float('inf') for _ in range(gameState.getFood().height)] for _ in range(gameState.getFood().width)]
    
    for (i, ghost) in enumerate(ghost_states):
        queue = util.Queue()
        queue.push((ghost.getPosition(), 0))
        visited = set()
        while not queue.isEmpty():
            current, distance = queue.pop()
            current = (int(current[0]), int(current[1]))
            print(current, distance)
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

