# Q learning
# Adapted from Thomas Simonini's tutorial
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb

import numpy as np
import random

import q_games
reload(q_games)

debug = True

# Default hyperparameters
hyperparameters = {
    "total_episodes":15000,         # Total episodes
    "learning_rate":0.8,         # Learning rate
    "max_steps":99,               # Max steps per episode
    "gamma":0.95,                  # Discounting rate

    # Exploration parameters
    "max_epsilon":1.0,             # Exploration probability at start
    "min_epsilon":0.01,            # Minimum exploration probability 
    "decay_rate":0.005            # Exponential decay rate for exploration prob
}

# Experimental hyperparameters
alt_hyperparameters = {
    "total_episodes":1000,         # Total episodes
    # A lower learning rate is necessary when there is randomization in the rewards
    "learning_rate":0.3,         # Learning rate
    "max_steps":100,               # Max steps per episode
    "gamma":0.95,                  # Discounting rate

    # Exploration parameters
    "max_epsilon":1.0,             # Exploration probability at start
    # For sufficiently simple games, setting min_epsilon=max_epsilon keeps play random and seems to work
    "min_epsilon":1.0,            # Minimum exploration probability 
    "decay_rate":0.005            # Exponential decay rate for exploration prob
}

def build_q_table(game):
    # Algorithm
    qtable = np.zeros((game["state_size"], game["action_size"]))
    epsilon = 1.0                 # Exploration rate
    for episode in range(hyperparameters["total_episodes"]):
        # Reset the environment
        state = game["reset"]()
        step = 0
        done = False
        if (debug and episode%100 == 0):
            print episode
        
        for step in range(hyperparameters["max_steps"]):
            exp_exp_tradeoff = random.uniform(0, 1)
            
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])
            else:
                action = int(random.uniform(0,game["action_size"]))
            new_state, reward, done = game["perform_action"](state,action)
        
            # Bellman function
            qtable[state, action] = qtable[state, action] + hyperparameters["learning_rate"] * (reward + hyperparameters["gamma"] * np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state
            if done == True: 
                break
        epsilon = hyperparameters["min_epsilon"] + (hyperparameters["max_epsilon"] - hyperparameters["min_epsilon"])*np.exp(-hyperparameters["decay_rate"]*episode) 
    return qtable
    
def play_game(game, qtable):
    state = game["reset"]()
    step = 0
    done = False
    score = 0
    moves = []

    for step in range(hyperparameters["max_steps"]):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        moves.append(action)
        new_state, reward, done = game["perform_action"](state,action)
        score += reward
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            print score
            
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
    return moves
    
# Change q_games.frozen_lake to another game if desired.
q = build_q_table(q_games.frozen_lake)
moves = play_game(q_games.frozen_lake, q)
print moves

