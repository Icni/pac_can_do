"""
This defines a Q-learning agent that performs a kind of Q-learning for the
purposes of training a model before the contest.
"""

import util
import random
import math
import pickle

from contest.capture_agents import CaptureAgent


x = False


def magnitude(v):
    return math.sqrt((v[0] * v[0]) + (v[1] * v[1]))


def dot(v1, v2):
    return (v1[0] * v2[0]) + (v1[1] * v2[1])


class ApproximateState:
    precision = 1

    def __init__(self, game_state, agent):
        agent_pos = game_state.get_agent_position(agent.index)
        enemies = game_state.get_blue_team_indices() if agent.red else game_state.get_red_team_indices()
        dists = game_state.get_agent_distances()

        self.enemy_vectors = []
        for enemy in enemies:
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos is not None:
                # Find (angle of enemy, distance of enemy)
                self.enemy_vectors.append((
                        round(math.acos(
                            min(dot(enemy_pos, agent_pos)
                                / (magnitude(enemy_pos) * magnitude(agent_pos)), 1)
                        ), self.precision + 1), dists[enemy]))

        self.min_team_dist = None
        for ally in agent.get_team(game_state):
            ally_pos = game_state.get_agent_position(ally)
            if ally_pos is not None:
                dist = round(agent.get_maze_distance(agent_pos, ally_pos), self.precision)
                if self.min_team_dist is None or dist < self.min_team_dist:
                    self.min_team_dist = dist

    def data(self):
        return (self.min_team_dist, self.enemy_vectors)

    def __hash__(self):
        return hash((self.min_team_dist, tuple(self.enemy_vectors)))

    def __eq__(self, o):
        return self.data() == o.data()


class QLearningAgent(CaptureAgent):
    q_values = util.Counter()

    def __init__(self, index, time_for_computing=.1,
                 epsilon=0.3, gamma=0.8, alpha=0.1, num_training=0):
        super().__init__(index, time_for_computing)
        self.start = None

        self.last_action = None

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.num_training = num_training

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        self.last_action = None
        self.load_q_values()
        print(len(self.q_values), end=', ')

    def get_q_value(self, state, action):
        """
        Q(state,action)
        """
        # approx = ApproximateState(state, self)
        return self.q_values[(state, action)]

    def compute_value_from_q_values(self, state):
        """
        max_(action in actions) Q(state,action)
        """
        max_q = None

        for action in state.get_legal_actions(self.index):
            q = self.get_q_value(state, action)
            if max_q is None or q > max_q:
                max_q = q

        return max_q or 0.0

    def compute_action_from_q_values(self, state):
        """
        Compute best action to take solely based on Q-values.
        Returns None if there are no legal actions to take.
        """
        max_q = None
        max_action = None

        for action in state.get_legal_actions(self.index):
            q = self.get_q_value(state, action)
            if max_q is None or q > max_q:
                max_q = q
                max_action = action
            elif q == max_q:
                max_action = random.choice((action, max_action))

        return max_action

    def get_action(self, state):
        """
        Compute best action to take from current state, taking a random action
        with probability self.epsilon.
        Returns None if there are no legal actions to take.
        """

        legal_actions = state.get_legal_actions(self.index)

        if len(legal_actions) == 0:
            self.last_action = None
        elif util.flip_coin(self.epsilon):
            self.last_action = self.reflexive_action(state)
        else:
            self.last_action = self.compute_action_from_q_values(state)

        if self.last_action is not None:
            successor = self.get_successor(state, self.last_action)

            food_multiplier = 1
            pos = successor.get_agent_position(self.index)
            if ((self.red and successor.get_red_food()[pos[0]][pos[1]])
                    or (not self.red and successor.get_blue_food()[pos[0]][pos[1]])):
                food_multiplier = 2

            reward = food_multiplier

            self.update(state,
                        self.last_action,
                        successor,
                        reward)

            if successor.is_over():
                self.save_q_values()

        return self.last_action

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # print(reward)
        max_next_q = self.compute_value_from_q_values(next_state)

        new_q = (((1 - self.alpha) * self.get_q_value(state, action))
                 + self.alpha * (reward + self.gamma * max_next_q))

        # approx = ApproximateState(state, self)
        self.q_values[(state, action)] = new_q

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)

    def reflexive_action(self, game_state):
        """
        `choose_action` from the demo code.
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = - \
            len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food)
                               for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def save_q_values(self):
        with open('q_dict.pkl', 'wb+') as f:
            pickle.dump(self.q_values, f)

    def load_q_values(self):
        try:
            with open('q_dict.pkl', 'rb') as f:
                self.q_values = pickle.load(f)
                # global x
                # if not x:
                #     print(self.q_values)
                #     x = True

        except FileNotFoundError:
            pass
