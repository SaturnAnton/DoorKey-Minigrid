import torch
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
from DQN import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
from torch import nn
import os
from datetime import datetime, timedelta

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:

    def preprocess(self, obs):
        img_flat  = obs['image'].flatten() / 255.0        
        direction = np.array([obs['direction']])
        return torch.tensor(np.concatenate([img_flat, direction]), dtype=torch.float, device=device)

    # In questa funzione passo gli iperparametri
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as f:
            all_hp = yaml.safe_load(f)
            hp     = all_hp[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set  # necessario per LOG/MODEL/GRAPH_FILE

        self.replay_memory_size = hp['replay_memory_size']
        self.mini_batch_size    = hp['mini_batch_size']
        self.epsilon_init       = hp['epsilon_init']
        self.epsilon_decay      = hp['epsilon_decay']
        self.epsilon_min        = hp['epsilon_min']
        self.network_sync_rate  = hp['network_sync_rate']
        self.discount_factor_g  = hp['discount_factor_g']
        self.learning_rate_a    = hp['learning_rate_a']
        self.enable_double_dqn  = hp['enable_double_dqn']

        self.loss_fn   = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')


    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human" if render else None)
        print(env.unwrapped.max_steps)
        env = FullyObsWrapper(env)

        num_states  = 193
        num_actions = env.action_space.n

        reward_per_episode = []
        epsilon_history    = []

        # Rete di policy
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory  = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            best_reward = -9999999

            # Rete di destinazione
            target_dqn = DQN(num_states, num_actions).to(device)

            # Sincronizzare la rete di destinazione in modo tale che abbia
            # la stessa struttura della rete di policy
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        else:
            # Carica la policy salvata e imposta la rete in modalità valutazione
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            obs, _ = env.reset()
            state  = self.preprocess(obs)

            terminated     = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # Andiamo a disattivare il calcolo del gradiente (che viene fatto in automatico)
                    # non stiamo facendo training, stiamo solo valutando uno stato
                    with torch.no_grad():
                        # unsqueeze = permette di aggiungere una dimensione aggiuntiva al tensore
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # item() = permette di ottenere il valore del tensore
                new_obs, reward, terminated, truncated, info = env.step(action.item())

                new_state = self.preprocess(new_obs)
                reward    = torch.tensor(reward, dtype=torch.float, device=device)

                episode_reward += reward.item()

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            # Salva il modello quando si ottiene un nuovo reward massimo
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:.2f} at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Aggiorna il grafico ogni 10 secondi
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Si entra in questo if se è stata accumulata abbastanza esperienza
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Permette di copiare la rete di policy nella rete di destinazione dopo un certo numero di passi
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # DE-STRUTTURAZIONE DEL BATCH
        # mini_batch è una lista di tuple (s, a, s', r, d). zip(*...) separa queste tuple in 5 liste indipendenti.
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # CREAZIONE DEI TENSORI (BATCH PROCESSING)
        # Trasformiamo le liste in tensori PyTorch per elaborarli massivamente sulla GPU/CPU.
        # states diventerà un tensore di forma [batch_size, 193]
        states = torch.stack(states)

        # actions diventerà [batch_size]
        actions = torch.stack(actions)

        # new_states diventerà [batch_size, 193]
        new_states = torch.stack(new_states)

        # rewards diventerà [batch_size]
        rewards = torch.stack(rewards)

        # Trasformiamo i booleani True/False in 1.0/0.0
        terminations = torch.tensor(terminations).float().to(device)

        # CALCOLO DEL TARGET
        # Usiamo torch.no_grad() perché non dobbiamo addestrare la rete 'target_dqn'
        with torch.no_grad():
            if self.enable_double_dqn:
                # [DOUBLE DQN] - SELEZIONE: la policy_dqn sceglie quale azione è la migliore
                # negli stati successivi (argmax dei Q-values della policy).
                # Questo disaccoppia la scelta dell'azione dalla sua valutazione.
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                # [DOUBLE DQN] - VALUTAZIONE: la target_dqn valuta il Q-value
                # dell'azione scelta dalla policy, riducendo l'overestimation bias.
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                    target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # DQN STANDARD - la rete target sceglie E valuta la stessa azione (max),
                # il che può portare a sovrastimare i Q-values.
                # Chiediamo alla rete target: "Qual è il valore massimo possibile nello stato successivo?"
                # .max(dim=1) restituisce (valori, indici). Noi prendiamo solo i valori [0].
                target_q_max = target_dqn(new_states).max(dim=1)[0]

                # EQUAZIONE DI BELLMAN:
                # Il valore target è la ricompensa immediata + il valore futuro scontato (gamma).
                # Se l'episodio è finito (terminations=1), ignoriamo il futuro: (1-1) * futuro = 0.
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_q_max

        # CALCOLO DEI VALORI Q ATTUALI (LA PREDIZIONE DELLA RETE)
        # Passiamo gli stati attuali alla policy_dqn. Otteniamo i Q-values per TUTTE le azioni.
        # .gather() seleziona solo il valore Q dell'azione che l'agente ha effettivamente scelto.
        # unsqueeze(dim=1) serve per far combaciare le dimensioni per il gather.
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # CALCOLO DELLA LOSS
        # Confrontiamo quanto la rete pensava di ottenere (current_q) rispetto a quanto ha effettivamente ottenuto (target_q).
        loss = self.loss_fn(current_q, target_q)

        # OTTIMIZZAZIONE
        self.optimizer.zero_grad()  # Pulisce i gradienti del passo precedente
        loss.backward()             # Calcola il gradiente (in che direzione correggere i pesi)
        self.optimizer.step()       # Aggiorna i pesi della rete neurale


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Nome del set di iperparametri da usare (es. doorkey8x8)')
    parser.add_argument('--train', help='Modalità training', action='store_true')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        # Modalità training: l'agente impara dall'ambiente senza rendering per velocità
        agent.run(is_training=True, render=False)
    else:
        # Modalità test: carica il modello salvato e lo visualizza con rendering
        agent.run(is_training=False, render=True)