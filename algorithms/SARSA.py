import gymnasium as gym
from minigrid.wrappers import *
from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

env = gym.make("MiniGrid-DoorKey-5x5-v0",max_steps = 500)

EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
TEMP = 1.0

def save_file(q):
    dict(**q)
    folder_path = os.path.join("..","data", "sarsa")
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, "sarsa15.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(dict(**q), file)

def get_state_key(obs):
    img_flat = obs['image'][:, :, 0].flatten()  # Prende il primo strato della matrice e restituisce una lista monodimensionale
    direction = obs['direction']
    return str(list(img_flat) + [direction])

def epsilon_greedy(q, state, epsilon, n_actions):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # Sceglie un'azione a caso tra tutte quelle disponibili
    else:
        # Per ogni azione possibile, va a leggere nella q-table il valore assegnato dall'agente e prende il valore più alto
        q_values = q[state]
        max_val = np.max(q_values)

        # Lista delle azioni migliori (nel caso ci dovessero essere pari merito)
        best_actions = [a for a in range(n_actions) if q_values[a] == max_val]

        return np.random.choice(best_actions)

def softmax(q, state,temp,n_actions):
    e = np.exp(q[state] / temp)
    return np.random.choice(n_actions, p=e / e.sum())
    
def sarsa(environment, episodes, alpha, gamma, expl_func, expl_param):
    n_actions = environment.action_space.n
    q = defaultdict(lambda: np.zeros(n_actions))
    rews = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for i in range(episodes):
        step = 0
        rewards = 0
        observation, info = environment.reset()
        s = get_state_key(observation)
        a = expl_func(q, s, expl_param, n_actions)
        done = False

        while not done:
            obs_next, r, terminated, truncated, _ = environment.step(a)
            s1 = get_state_key(obs_next)
            done = terminated or truncated
            if(terminated):
                print("Obiettivo raggiunto")
            custom_reward = r

            a1 = expl_func(q,s1,expl_param,n_actions)

            # Equazione di Bellman per l'aggiornamento del valore Q(s, a)
            q[s][a] = q[s][a] + alpha * (custom_reward + gamma * q[s1][a1] - q[s][a])

            s = s1
            a = a1
            step += 1
            rewards += custom_reward

        rews[i] = rewards
        lengths[i] = step
        
        if(expl_param > 0.1):
            expl_param = expl_param * 0.995

        print(f"siamo all'episodio {i}")
    
    return q, rews, lengths

print("Inizio del training...")
sol, rews, lengths = sarsa(env, EPISODES, ALPHA, GAMMA, softmax, TEMP)
print("Fine")

# Creiamo un grafico con due "sotto-grafici"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Grafico 1: Ricompense
# Calcoliamo una media mobile per rendere la curva più dolce e leggibile
window = 50
smoothed_rews = np.convolve(rews, np.ones(window)/window, mode='valid')
ax1.plot(smoothed_rews, color='green')
ax1.set_title('Ricompense nel tempo (Media mobile)')
ax1.set_xlabel('Episodi')
ax1.set_ylabel('Ricompensa Totale')

# Grafico 2: Lunghezza episodi (Passi)
smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
ax2.plot(smoothed_lengths, color='blue')
ax2.set_title('Passi per Episodio (Media mobile)')
ax2.set_xlabel('Episodi')
ax2.set_ylabel('Numero di Passi')

plt.tight_layout()
plt.show()

save_file(sol)