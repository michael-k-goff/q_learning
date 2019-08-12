# Games for Q learning

# Each game has action_size (number of actions that can be performed) and state_size (number of states)
# Each game has a perform_action, which takes a state,action pair and returns new_state, reward, is_done

import gym
import random

# Navigate a 5X5 maze. Some squares are blocked, others give rewards.
def perform_action_maze(state,action):
    # For action, treat 0,1,2,3 as left, right, down, up
    # For state, state i is the position (i%5, i//5), where the upper left corner is (0,0).

    # Basic rules: lose a point for every move, successful or now.
    # The game ends in the lower left or upper right corners with 50 points, or the lower right corner with 100 points.
    # Cannot enter tiles where both coordinates are odd.
    x,y = state%5, state//5
    new_x = x + [-1,1,0,0][action]
    new_y = y + [0,0,1,-1][action]
    if (new_x < 0 or new_y < 0 or new_x >= 5 or new_y >= 5):
        new_x, new_y = x,y
    if (new_x*new_y)%2:
        new_x, new_y = x,y
    new_state = new_x + new_y*5
    if (new_x == 0 and new_y == 4):
        return new_state, 50, True
    if (new_x == 4 and new_y == 0):
        return new_state, 50, True
    if (new_x == 4 and new_y == 4):
        return new_state, 100, True
    return new_state,-1,False
maze = {
    "action_size":4,
    "state_size":25,
    "perform_action":perform_action_maze,
    "reset":lambda: 0
}

# Las Vegas Strip. There are three actions: 0: walk left, 1: walk right, 2: gamble
# The reward is random and depends on the position on the strip.
# The game never ends
# This game illustrates that Q Learning can find a good strategy when the game has randomness.
win_odds = [0.8,0.6,0.4,0.2,0.1]
losses = [-1,-2,-3,-4,-5]
wins = [0,1,10,6,10]
def perform_action_strip(state,action):
    if action<2:
        new_state = state
        if action == 0:
            new_state = max(new_state-1,0)
        if action == 1:
            new_state = min(4,new_state+1)
        return new_state,0,False
    else:
        val = random.random()
        if val < win_odds[state]:
            return state,wins[state],False
        else:
            return state,losses[state],False
strip = {
    "action_size":3,
    "state_size":5,
    "perform_action":perform_action_strip,
    "reset":lambda: 0
}

# Frozen Lake from OpenAI Gym.
# The following is a wrapper so it can be used by q.py.
env = gym.make("FrozenLake-v0")
def frozen_lake_action(state,action):
    new_state, reward, done, info = env.step(action)
    return new_state, reward, done
frozen_lake = {
    "action_size":env.action_space.n,
    "state_size":env.observation_space.n,
    "perform_action":frozen_lake_action,
    "reset":lambda: env.reset()
}

