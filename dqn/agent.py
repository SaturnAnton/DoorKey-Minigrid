import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from DQN import DQN
from datetime import datetime, timedelta
import itertools
import argparse
import minigrid
from minigrid.wrappers import FullyObsWrapper
import os

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR    = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_state_tensor(obs):
    img_flat  = obs['image'].flatten().astype(np.float32)
    direction = np.array([obs['direction']], dtype=np.float32)
    return np.concatenate([img_flat, direction])


class Agent:

    def __init__(self, hyperparameter_set, fully_observable=None):
        with open('hyperparameters.yml', 'r') as f:
            all_hp = yaml.safe_load(f)
            hp     = all_hp[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id             = hp['env_id']
        self.learning_rate_a    = hp['learning_rate_a']
        self.discount_factor_g  = hp['discount_factor_g']
        self.network_sync_rate  = hp['network_sync_rate']
        self.replay_memory_size = hp['replay_memory_size']
        self.mini_batch_size    = hp['mini_batch_size']
        self.epsilon_init       = hp['epsilon_init']
        self.epsilon_decay      = hp['epsilon_decay']
        self.epsilon_min        = hp['epsilon_min']
        self.stop_on_reward     = hp['stop_on_reward']
        self.fc1_nodes          = hp['fc1_nodes']
        self.max_episode_steps  = hp.get('max_episode_steps', 500)
        self.env_make_params    = hp.get('env_make_params', {})

        if fully_observable is not None:
            self.fully_observable = fully_observable
        else:
            self.fully_observable = hp.get('fully_observable', False)

        self.loss_fn   = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def _make_env(self, render=False):
        """Crea l'ambiente, applicando FullyObsWrapper se richiesto."""
        env = gym.make(
            self.env_id,
            render_mode='human' if render else None,
            max_episode_steps=self.max_episode_steps,
            **self.env_make_params
        )
        if self.fully_observable:
            env = FullyObsWrapper(env)
        return env

    def run(self, is_training=True, render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as f:
                f.write(log_message + '\n')

        env = self._make_env(render=render)

        # Dimensione stato: tutti e 3 i canali + direzione
        obs_shape   = env.observation_space['image'].shape   # (H, W, 3)
        num_states  = obs_shape[0] * obs_shape[1] * obs_shape[2] + 1
        num_actions = env.action_space.n

        print(f"State size: {num_states} | Actions: {num_actions} | "
              f"FullyObs: {self.fully_observable} | Device: {device}")

        rewards_per_episode = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training:
            epsilon         = self.epsilon_init
            memory          = ReplayMemory(self.replay_memory_size)
            target_dqn      = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer  = torch.optim.Adam(policy_dqn.parameters(),
                                               lr=self.learning_rate_a)
            epsilon_history = []
            step_count      = 0
            best_reward     = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        for episode in itertools.count():
            obs, _         = env.reset()
            state          = torch.tensor(get_state_tensor(obs),
                                          dtype=torch.float, device=device)
            terminated     = False
            truncated      = False
            episode_reward = 0.0

            while not terminated and not truncated and episode_reward < self.stop_on_reward:

                # epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                new_obs, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(get_state_tensor(new_obs),
                                         dtype=torch.float, device=device)
                reward_t  = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward_t, terminated))
                    step_count += 1

                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if episode_reward > best_reward:
                    log_message = (
                        f"{datetime.now().strftime(DATE_FORMAT)}: "
                        f"New best reward {episode_reward:.1f} at episode {episode}, "
                        f"epsilon {epsilon:.4f} — saving model..."
                    )
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if datetime.now() - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = datetime.now()
                
                if episode % 100 == 0:
                    print(f"siamo all'episodio {episode}")

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.array([
            np.mean(rewards_per_episode[max(0, x - 99): x + 1])
            for x in range(len(rewards_per_episode))
        ])

        plt.subplot(121)
        plt.ylabel('Mean Rewards (last 100 ep.)')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states       = torch.stack(states)
        actions      = torch.stack(actions)
        new_states   = torch.stack(new_states)
        rewards      = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                       target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(1)
        ).squeeze()

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test DQN agent.')
    parser.add_argument('hyperparameters',
                        help='Nome del set di iperparametri nel yaml')
    parser.add_argument('--train',
                        action='store_true',
                        help='Modalita training')
    parser.add_argument('--no-fully-obs',
                        action='store_true',
                        help='Disabilita FullyObsWrapper (usa vista parziale 7x7)')
    args = parser.parse_args()

    fully_observable_override = False if args.no_fully_obs else None

    agent = Agent(
        hyperparameter_set=args.hyperparameters,
        fully_observable=fully_observable_override
    )

    if args.train:
        agent.run(is_training=True, render=False)
    else:
        agent.run(is_training=False, render=True)