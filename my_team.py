import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


SEARCH_DEPTH = 5


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensivePlanningAgent', second='DefensivePlanningAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class MinimaxNode:
    def __init__(self, state, minimax_agent, agent_index=None, depth=0, alpha=None, beta=None):
        self.alpha = alpha if alpha is not None else float('-inf')
        self.beta = beta if beta is not None else float('inf')

        self.state = state
        self.agent = minimax_agent
        self.action_state_pairs = {}
        # Standard: we start with index from each agent
        self.agent_index = minimax_agent.index if agent_index is None else agent_index
        self.depth = depth
        self.cut = False

        self.best_action = None
        self.value = float('inf')
        if self.is_max():
            self.value = float('-inf')

        actions = state.get_legal_actions(self.agent_index)

        if self.is_leaf() or len(actions) == 0:
            self.value = self.agent.utility_function(self.state)
        else:
            for action in actions:
                if self.cut:
                    break
                self.expand_node(action)

    def next_agent_index(self):
        index = self.agent_index
        while self.state.get_agent_state(self.agent_index).configuration is None:
            index = (index + 1) % self.state.get_num_agents()

        return index

    def expand_node(self, action):
        successor = self.state.generate_successor(self.agent_index, action)

        next_agent_index = self.next_agent_index()

        next_depth = self.depth
        # Once we hit max, we go to next depth
        if next_agent_index == self.agent.index:
            next_depth += 1

        node = self.child(successor, next_agent_index, next_depth)
        self.action_state_pairs[action] = node

        self.update_action_value(node, action)

    def update_action_value(self, node, action):
        if self.is_max():
            if self.value > self.beta:
                self.cut = True
            if node.value > self.value:
                self.value = node.value
                self.best_action = action
        else:
            if self.value < self.alpha:
                self.cut = True
            if node.value < self.value:
                self.value = node.value
                self.best_action = action

    def is_max(self):
        return (self.state.is_on_red_team(self.agent_index)
                == self.state.is_on_red_team(self.agent.index))

    def is_min(self):
        return not self.is_max()

    def is_leaf(self):
        return self.depth >= self.agent.depth

    def child(self, state, agent_index, depth):
        next_alpha = self.alpha
        if self.is_max():
            next_alpha = max(self.alpha, self.value)

        next_beta = self.beta
        if self.is_min():
            next_beta = min(self.beta, self.value)

        return MinimaxNode(state, self.agent, agent_index, depth, next_alpha, next_beta)


class PlanningCaptureAgent(CaptureAgent):
    """
    Planning agent, plans to maximize own team (since our ability to observe
    pacman is quite limited, we can't do a proper minimax) (we tried).
    """

    def __init__(self, index, time_for_computing=.1, depth=SEARCH_DEPTH):
        super().__init__(index, time_for_computing)
        self.start = None
        self.depth = depth
        self.index = index
        self.max_team = None

    def is_red(self):
        return self.max_team == 0

    def is_blue(self):
        return self.max_team == 1

    def register_initial_state(self, game_state):
        self.max_team = 0 if game_state.is_on_red_team(self.index) else 1
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def choose_action(self, game_state):
        tree = MinimaxNode(game_state, self)

        if tree.best_action is None:
            actions = game_state.get_legal_actions(self.index)
            if len(actions) == 0:
                return Directions.STOP
            return random.choice(actions)
        return tree.best_action

    def utility_function(self, game_state):
        features = self.get_state_features(game_state)
        weights = self.get_state_weights(game_state)
        return features * weights

    def get_state_features(self, game_state):
        feats = util.Counter()
        feats['score'] = self.get_score(game_state)
        return feats

    def get_state_weights(self, game_state):
        return {'score': 1.0}


class OffensivePlanningAgent(PlanningCaptureAgent):
    """
    Offensive Planning Agent:
    - Eats food from opponant
    - Wants to be close to food
    """

    def get_state_features(self, game_state):
        features = util.Counter()

        # Food from opponant
        food_list = self.get_food(game_state).as_list()
        features['successor_score'] = -len(food_list)

        # Distance to next food
        if len(food_list) > 0:
            my_pos = game_state.get_agent_state(self.index).get_position()
            min_distance = min(self.get_maze_distance(my_pos, food)
                               for food in food_list)
            features['distance_to_food'] = min_distance

        # Score at the moment
        features['score'] = self.get_score(game_state)

        enemies = [(game_state.get_agent_state(i), i)
                   for i in self.get_opponents(game_state)]

        # Computes distance to nearest enemy (using rough distance if needed)
        dists = []
        for a, i in enemies:
            if a.get_position() is not None:
                d = self.get_maze_distance(my_pos, a.get_position())
            else:
                d = game_state.get_agent_distances()[i]
            d *= a.scared_timer
            dists.append(d)
        if len(dists) > 0:
            features['enemy_distance'] = min(dists)

        # Sum of scared timers
        features['num_scared'] = sum([a.scared_timer for (a, i) in enemies])

        return features

    def get_state_weights(self, game_state):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'score': 10,
            'num_scared': 5,
            'enemy_distance': 100,
        }


class DefensivePlanningAgent(PlanningCaptureAgent):
    """
    Defense Planning Agent:
    - Wants to be close to invading pacmans
    - Wants to be close to food if there are no invaders
    """

    def get_state_features(self, game_state):
        features = util.Counter()

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Finds invading pacmen
        enemies = [(game_state.get_agent_state(i), i)
                   for i in self.get_opponents(game_state)]
        invaders = [(a, i) for (a, i) in enemies if a.is_pacman]
        features['num_invaders'] = len(invaders)

        # Computes distance to nearest invader (using rough distance if needed)
        dists = []
        for a, i in invaders:
            if a.get_position() is not None:
                dists.append(self.get_maze_distance(my_pos, a.get_position()))
            else:
                d = game_state.get_agent_distances()
                dists.append(d[i])
        if len(dists) > 0:
            features['invader_distance'] = min(dists)

        features['distance_to_food'] = 0
        if len(invaders) == 0:
            food_list = self.get_food(game_state).as_list()

            # Distance to next food
            if len(food_list) > 0:
                my_pos = game_state.get_agent_state(self.index).get_position()
                min_distance = min(self.get_maze_distance(my_pos, food)
                                   for food in food_list)
                features['distance_to_food'] = min_distance

        return features

    def get_state_weights(self, game_state):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -100,
            'distance_to_food': -1,
        }
