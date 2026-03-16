import gymnasium as gym
from minigrid.wrappers import *
from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

env = gym.make("MiniGrid-DoorKey-5x5-v0",max_steps = 500)

EPISODES = 5000  # EPISODI = numero di tentativi che l'agente fa per imparare
ALPHA = 0.1      # LEARNING RATE = velocità di apprendimento. Minore è il valore, maggiore è la memoria storica
GAMMA = 0.99     # DISCOUNT FACTOR = valore che indica quanto l'agente ci tiene al futuro
EPSILON = 1.0   # EPSILON = probabilità di eseguire una mossa casuale

def save_file(q):
    dict(**q)
    os.makedirs("q_table", exist_ok=True)
    filepath = os.path.join("q_table", "qtable10.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(dict(**q), file)

# Funzione che unisce l'immagine e la direzione trasformandole in una tupla
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


def q_learning(environment, episodes, alpha, gamma, expl_func, expl_param):
    # Numero totale di azioni disponibili nell'environment
    n_actions = environment.action_space.n

    # La q-table mappa ogni stato a un array di valori per ogni azione possibile.
    # Usiamo defaultdict con zeros per inizializzare automaticamente stati mai visti.
    q = defaultdict(lambda: np.zeros(n_actions))

    rews = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for i in range(episodes):
        step = 0
        rewards = 0
        observation, info = environment.reset()

        s = get_state_key(observation)
        done = False

        while not done:
            # Applicazione di epsilon greedy per scegliere l'azione tra tutte quelle dell'environment
            a = expl_func(q, s, expl_param, n_actions)

            # L'azione scelta viene eseguita nell'ambiente
            obs_next, r, terminated, truncated, _ = environment.step(a)
            s1 = get_state_key(obs_next)
            if(terminated):
                print("Obiettivo raggiunto")
            done = terminated or truncated

            custom_reward = r

            # Aggiornamento di Bellman: il valore massimo futuro viene calcolato
            # su tutte le azioni disponibili nell'environment
            best_next_action_val = np.max(q[s1])

            # Equazione di Bellman per l'aggiornamento del valore Q(s, a)
            q[s][a] = q[s][a] + alpha * (custom_reward + gamma * best_next_action_val - q[s][a])

            s = s1
            step += 1
            rewards += custom_reward

        rews[i] = rewards
        lengths[i] = step
        if(expl_param > 0.1):
            expl_param = expl_param * 0.995
    

        print (f"siamo al {i} episodio")

    return q, rews, lengths


sol, rews, lengths = q_learning(env, EPISODES, ALPHA, GAMMA, epsilon_greedy, EPSILON)
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

