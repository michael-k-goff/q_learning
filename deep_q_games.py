# Games for Deep Q Learning

# To run: open Python3 shell and exec(open("deep_q_games.py").read())

import random
import numpy as np

# Parameters: action_size is the number of possible actions
# state_size is the dimensionality of the state space

# The following is a very simple game with five continuous parameters that define the states
# The first two parameters are "knobs" that are directly adjusted by input.
# The remaining parameters updated indirectly based on previous parameters. The goal is to maximize the last parameters over time.
# The optimum strategy is not obvious (at least not to me).
class Game:
    def __init__(self):
        self.action_size = 4
        self.state_size = 5
        self.reset()
    def reset(self):
        self.internal_data = np.zeros([5])
        return self.internal_data
    def perform_action(self,action):
        if action[0]:
            self.internal_data[0] = max(0,self.internal_data[0]-0.1)
        if action[1]:
            self.internal_data[0] = (self.internal_data[0]+0.1)%1
        if action[2]:
            self.internal_data[1] = max(0,self.internal_data[1]-0.1)
        if action[3]:
            self.internal_data[1] = (self.internal_data[1]+0.1)%1
        self.internal_data[2] = (self.internal_data[2] + 0.1*self.internal_data[0])%1
        self.internal_data[3] = (self.internal_data[3] + 0.1*self.internal_data[1])%1
        self.internal_data[4] = (self.internal_data[4] + 0.1*self.internal_data[2]*self.internal_data[3])%1
        return self.internal_data, self.internal_data[4], False
    def get_state(self):
        return self.internal_data
