from frozen_lake import FrozenLakeEnvGui
from main import value_iteration


def main():
    env = FrozenLakeEnvGui()
    V, pi = value_iteration(env)
    env.reset_gui()
    while True:
        obs = env.reset()
        done = False
        rew = 0.0
        env.render_gui(V, is_q_value=False)
        while not done:
            action = pi[obs]
            obs, rew, done, _ = env.step(action)
            env.render_gui(V, is_q_value=False)
        print(f"Reward: {rew}")


if __name__ == '__main__':
    main()
