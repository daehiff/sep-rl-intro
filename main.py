from pprint import pprint
from time import sleep

import gym
import numpy as np

from frozen_lake import FrozenLakeEnvGui


def choose_action(env, Q, observation, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[observation, :])
    return action


def q_learning(env, episodes=500, epsilon=0.9, epsilon_decay=0.95, min_epsilon=0.01, gamma=0.95, lr=0.81):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for ep in range(episodes):
        done = False
        obs = env.reset()
        print(f"Episode: {ep}")
        while not done:
            env.render_gui(Q)

            action = choose_action(env, Q, obs, epsilon)
            obs2, rew, done, _ = env.step(action)

            prediction = Q[obs, action]
            target = rew + gamma * np.max(Q[obs2, :])
            Q[obs, action] = Q[obs, action] + lr * (target - prediction)
            obs = obs2
        env.render_gui(Q)
        if ep % 100 == 0:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, min_epsilon)
    return Q


def value_iteration(env, gamma=0.95, omega=0.1):
    V = [0.0 for _ in range(env.nS)]
    ep = 0.0
    while True:
        print(f"Episode: {ep}")
        ep += 1
        delta = 0.0

        env.render_gui(V, is_q_value=False)
        for state in range(env.nS):
            v_ = float(V[state])
            action_values = []
            for action in range(env.nA):
                action_value = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    action_value += prob * (reward + gamma * V[next_state])
                action_values.append(action_value)
            best_action = np.max(np.asarray(action_values))
            V[state] = float(best_action)
            delta = max(delta, abs(V[state] - v_))
        if delta < omega:
            break
    pi = np.zeros((env.nS))
    for state in range(env.nS):
        pi[state] = convert_value_to_policy(env, V, state, gamma)
    return V, pi


def convert_value_to_policy(env, V, obs, gamma):
    adjacent_states = env.P[obs]
    action_values = []
    for movement, state in adjacent_states.items():
        action_value = 0.0
        for prob, next_state, reward, done in state:
            action_value += prob * (reward + gamma * V[next_state])
        action_values.append(action_value)
    return np.argmax(action_values)


def main(render=True):
    env = FrozenLakeEnvGui()
    Q = q_learning(env)
    # env.reset()
    # V = value_iteration(env)


if __name__ == '__main__':
    main()
