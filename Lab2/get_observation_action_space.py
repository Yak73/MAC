import gym
from gym.spaces import *
import sys

GAME_NAME = 'Alien-v0'
PRINT_ALL = True


def print_spaces(space):
    print(space)
    if isinstance(space, Box) and PRINT_ALL:
        print("\n space.low: ", space.low)
        print("\n space.high: ", space.high)


if __name__ == '__main__':
    env = gym.make(GAME_NAME)
    print("Observation Space:")
    print_spaces(env.observation_space)
    print("Action Space:")
    print_spaces(env.action_space)
    try:
        print("Action description/meaning: ", env.unwrapped.get_action_meanings())
    except AttributeError:
        pass
    print_spaces(env.state)
