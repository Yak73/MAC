# source code - https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pyautogui as pyag


# Load the environment
env = gym.make('Taxi-v2')  # make function of Gym loads the specified environment

s_reset = env.reset()  # resets the environment and returns the start state as a value
print("Initial state : ", s_reset)
env.render()
print("\nNumber of actions: ", env.action_space)
print("number of states: ", env.observation_space, "\n")


# Epsilon-Greedy approach for Exploration and Exploitation of the state-actions spaces
def epsilon_greeddy(Q, s, _, epsilon=0.3):
    p = np.random.uniform(low=0, high=1)
    if p > epsilon:
        return np.argmax(Q[s, :])  # say here,initial policy = for each state consider the action having highest
    else:
        return env.action_space.sample()


"""
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""


def learn(q_table, state, state2, reward, action, lr, df):
    q_table[state, action] = q_table[state, action] + lr * \
                             (reward + df * np.max(q_table[state2, :]) - q_table[state, action])
    return q_table


# Q-Learning Implementation
def q_learning(lr, df, eps, show_table=True):
    epsilon = 0.3
    # Initializing Q-table with zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for i in range(eps):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greeddy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, info = env.step(action)
            Q = learn(Q, state, next_state, reward, action, lr, df)
            state = next_state

    if show_table:
        print(Q)

    return Q


def check_success(Q, random=False):
    s = env.reset()
    # env.render()
    # print(s)
    while True:
        if random:
            a = int(np.random.uniform(low=0, high=6))
        else:
            a = np.argmax(Q[s])
        s_, r, t, _ = env.step(a)
        # print("===============")
        # print(s_, r, t)
        # env.render()
        s = s_
        if t is True:
            break

    if r == 20:
        return True
    return False


def experiment(lr=0.5, df=0.9, eps=10000, number=10, rnd=False):
    sum_n, sum_time = 0, 0
    Q_zero = np.zeros((500, 6))
    for _ in range(number):
        start = time.time()
        success = False
        n = 0
        while not success:
            if rnd:
                Q = Q_zero
            else:
                Q = q_learning(lr=lr, df=df, eps=eps, show_table=False)
            success = check_success(Q, random=True)
            n += 1
        sum_n += n
        sum_time += time.time() - start
    return sum_n/number, sum_time/number


def analysis_params(random_bit=False):
    # Careful: with defaults params estimate time of work is 1 hour
    repeat_number = 1
    avg_results_for_success = list()

    avg_results_for_success.append(experiment(lr=0.5, df=0.9, eps=10000, number=repeat_number, rnd=random_bit))  # 0
    if random_bit:
        print(avg_results_for_success)
        return

    avg_results_for_success.append(experiment(lr=0.4, df=0.9, eps=10000, number=repeat_number, rnd=random_bit))  # 1
    avg_results_for_success.append(experiment(lr=0.6, df=0.9, eps=10000, number=repeat_number, rnd=random_bit))  # 2
    avg_results_for_success.append(experiment(lr=0.5, df=0.85, eps=10000, number=repeat_number, rnd=random_bit))  # 3
    avg_results_for_success.append(experiment(lr=0.5, df=0.95, eps=10000, number=repeat_number, rnd=random_bit))  # 4
    avg_results_for_success.append(experiment(lr=0.5, df=0.9, eps=2000, number=repeat_number, rnd=random_bit))  # 5
    avg_results_for_success.append(experiment(lr=0.5, df=0.9, eps=25000, number=repeat_number, rnd=random_bit))  # 6

    print(avg_results_for_success)

    f_lr_n = [avg_results_for_success[x][0] for x in (1, 0, 2)]
    f_lr_time = [avg_results_for_success[x][1] for x in (1, 0, 2)]
    x_lr = [0.4, 0.5, 0.6]

    f_df_n = [avg_results_for_success[x][0] for x in (3, 0, 4)]
    f_df_time = [avg_results_for_success[x][1] for x in (3, 0, 4)]
    x_df = [0.85, 0.9, 0.95]

    f_eps_n = [avg_results_for_success[x][0] for x in (5, 0, 6)]
    f_eps_time = [avg_results_for_success[x][1] for x in (5, 0, 6)]
    x_eps = [2000, 10000, 25000]

    fig, ax = plt.subplots(1, 3, squeeze=False)

    ax[0][0].plot(x_lr, f_lr_n, x_lr, f_lr_time, marker=".")
    ax[0][0].set_title("Learning rate")
    ax[0][0].legend(("suc_№", "suc_time"), loc='upper center')

    ax[0][1].plot(x_df, f_df_n, x_df, f_df_time, marker=".")
    ax[0][1].set_title("Discount factor")
    ax[0][1].legend(("suc_№", "suc_time"), loc='upper center')

    ax[0][2].plot(x_eps, f_eps_n, x_eps, f_eps_time, marker=".")
    ax[0][2].set_title("Total episodes")
    ax[0][2].legend(("suc_№", "suc_time"), loc='upper center')

    plt.show()


def clear_screen(cnsl=False):
    time.sleep(1)
    if cnsl:
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        pyag.click(x=197, y=782)
        pyag.hotkey('f11')


def view_one_pass():

    Q = q_learning(lr=0.5, df=0.9, eps=10000, show_table=False)
    # guarantee of success
    while not check_success(Q):
        Q = q_learning(lr=0.5, df=0.9, eps=10000, show_table=False)

    s = env.reset()

    while True:
        a = np.argmax(Q[s])
        s_, r, t, _ = env.step(a)
        clear_screen()
        env.render()
        s = s_
        if t is True:
            break


if __name__ == '__main__':
    print("Write "
          "\n1 for analysis_params  (work time ~ 20 min)"
          "\n0 for view_one_pass")
    choose = int(input())
    if choose:
        analysis_params(random_bit=False)
        analysis_params(random_bit=True)
    else:
        view_one_pass()


