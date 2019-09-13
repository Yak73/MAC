import gym
import numpy as np
import time

# Load the environment

env = gym.make('FrozenLake-v0')  # make function of Gym loads the specified environment

s_reset = env.reset()  # resets the environment and returns the start state as a value
print("Initial state : ", s_reset)
env.render()
print("\nNumber of actions: ", env.action_space)
print("number of states: ", env.observation_space, "\n")


# Epsilon-Greedy approach for Exploration and Exploitation of the state-actions spaces
def epsilon_greedy(Q, s, na):
    epsilon = 0.3
    p = np.random.uniform(low=0, high=1)
    if p > epsilon:
        return np.argmax(Q[s, :])  # say here,initial policy = for each state consider the action having highest
    else:
        return env.action_space.sample()


# Q-Learning Implementation
def q_learning(lr, y, eps, show_table=True):
    # Initializing Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(eps):
        s = env.reset()
        t = False
        while True:
            a = epsilon_greedy(Q, s, env.action_space.n)
            s_, r, t, _ = env.step(a)
            if r == 0:
                if t is True:
                    r = -5  # to give negative rewards when holes turn up
                    Q[s_] = np.ones(env.action_space.n)*r  # in terminal state Q value equals the reward
                else:
                    r = -1  # to give negative rewards to avoid long routes
            if r == 1:
                r = 100
                Q[s_] = np.ones(env.action_space.n)*r   # in terminal state Q value equals the reward
            Q[s, a] = Q[s, a] + lr * (r + y*np.max(Q[s_, a]) - Q[s, a])
            s = s_
            if t is True:
                break
    if show_table:
        print("Q-table\n{}\n".format(Q))

    return Q


def check_success(Q):
    s = env.reset()
    # env.render()
    # print(s)

    while True:
        a = np.argmax(Q[s])
        s_, r, t, _ = env.step(a)
        # print("===============")
        # print(s_, r, t)
        # env.render()
        s = s_
        if t is True:
            break
    if s == 15:
        return True


def experiment(lr=0.5, y=0.9, eps=10000):
    start = time.time()
    success = False
    n = 0
    while not success:
        Q = q_learning(lr=lr, y=y, eps=eps, show_table=False)
        success = check_success(Q)
        n += 1
    print("Params lr={}, y={}, eps={}   program time: {} seconds".format(lr, y, eps, str(time.time() - start)))



if __name__ == '__main__':
    experiment()


