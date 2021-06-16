import numpy as np


def choose_action(env, Q, observation, epsilon):
    """
    Chose a action either random or based on Q-Values
    :param env:
    :param Q:
    :param observation:
    :param epsilon:
    :return:
    """
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[observation, :])
    return action


def q_learning(env, episodes=500, epsilon=0.9, epsilon_decay=0.95, min_epsilon=0.01, gamma=0.95, lr=0.81):
    """
    Q-Learning implementation
    :param env: the openAI gym enviroment
    :param episodes: how many episodes of training
    :param epsilon: pr of random moves
    :param epsilon_decay: pr decay of random moves
    :param min_epsilon: min pr of random moves
    :param gamma: look ahead of algorithm
    :param lr: learning rate
    :return: Q, pi policy and corresponding Q-Values
    """
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
    pi = np.zeros((env.nS))
    for state in range(env.nS):
        pi[state] = np.argmax(Q[state])
    return Q, pi


def value_iteration(env, gamma=0.95, omega=0.1):
    """
    Value iteration implementation
    :param env: openAI gym env (in this case frozen-lake GUI)
    :param gamma: look a head value
    :param omega: determine when V-Values are fine enough
    :return: V, pi, V-Values and optimal policy
    """
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
    """
    Determine the policy from the V-Values
    :param env:
    :param V:
    :param obs:
    :param gamma:
    :return:
    """
    adjacent_states = env.P[obs]
    action_values = []
    for movement, state in adjacent_states.items():
        action_value = 0.0
        for prob, next_state, reward, done in state:
            action_value += prob * (reward + gamma * V[next_state])
        action_values.append(action_value)
    return np.argmax(action_values)
