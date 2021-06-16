from frozen_lake import FrozenLakeEnvGui
from main import q_learning


def main():
    env = FrozenLakeEnvGui()
    Q, pi = q_learning(env)
    env.reset_gui()
    while True:
        obs = env.reset()
        done = False
        rew = 0.0
        env.render_gui(Q, is_q_value=True)
        while not done:
            action = pi[obs]
            obs, rew, done, _ = env.step(action)
            env.render_gui(Q, is_q_value=True)
        print(f"Reward: {rew}")


if __name__ == '__main__':
    main()
