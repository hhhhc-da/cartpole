import torch
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output


class DQN:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=64, lr=0.001):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim * 2, action_dim)
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def update(self, state: np.ndarray, q: np.ndarray):
        q_pred = self.model(torch.Tensor(state))
        loss = self.criterion(q_pred, torch.Tensor(q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def plot_result(values, title=''):
    clear_output()
    f, ax = plt.subplots(1, 2, figsize=(12, 10))
    f.suptitle(title)

    ax[0].plot(values, label='reward per episode')
    ax[0].axhline(200, c='red', label='goal')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Reward')
    ax[0].legend()

    ax[1].set_title('mean reward : {}'.format(sum(values[-50:])/50))
    ax[1].hist(values[-50:])
    ax[1].axvline(200, c='red', label='goal')
    ax[1].set_xlabel('Reward per last 50 episode')
    ax[1].set_ylabel("Requency")
    ax[1].legend()

    plt.show()


def dqn_simple(env, model: DQN, episodes, gamma=0.9, epsilon=0.3, episilon_decay=0.9, title='dqn_simple'):
    rewards = []
    for i in range(episodes):
        print("Episode {}".format(i))
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            q_values = model.predict(state)
            q_values[action] = reward + (1 - int(done)) * \
                gamma * torch.max(model.predict(next_state)).item()

            model.update(state, q_values)
            state = next_state
        epsilon = max(epsilon * episilon_decay, 0.01)
        rewards.append(total_reward)
        # print('Episode {} - Reward = {}'.format(i, total_reward))

    plot_result(rewards)


if __name__ == "__main__":
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")

    simple_dqn = DQN(state_dim=4, action_dim=2, hidden_dim=64, lr=0.001)
    dqn_simple(env=env, model=simple_dqn, episodes=150,
               gamma=0.9, epsilon=0.5, episilon_decay=0.99)
