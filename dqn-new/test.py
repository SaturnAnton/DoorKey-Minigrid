import time
import torch
import numpy as np

from env import MinigridDoorKeyFullyObs
from model import MlpMinigridPolicy

def test_model(model_path, grid_size, num_episodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing su dispositivo: {device}")

    env = MinigridDoorKeyFullyObs(size=grid_size, render=True)
    num_actions = env.action_space.n
    state_space = env.observation_space.shape

    print(f"Caricamento dei pesi dal file: {model_path}")
    model = MlpMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_params'])
    
    model.eval()

    success_count = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        step_count = 0
        episode_reward = 0

        print(f"--- Inizio Episodio {episode + 1} ---")
        
        time.sleep(0.5)

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                q_values = model(state_tensor).cpu().numpy()
            
            action = np.argmax(q_values)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            state = next_state
            
            time.sleep(0.1)

        if episode_reward > 0:
            success_count += 1
            print(f"Risultato: VITTORIA in {step_count} passi! (Reward: {episode_reward:.3f})")
        else:
            print(f"Risultato: SCONFITTA/Timeout. (Passi: {step_count})")

    success_rate = (success_count / num_episodes) * 100
    print(f"\\n=== Riassunto Test ===")
    print(f"Episodi giocati: {num_episodes}")
    print(f"Vittorie: {success_count}")
    print(f"Success Rate: {success_rate:.1f}%")

    env.close()

if __name__ == "__main__":
    NOME_FILE_MODELLO = "8-8x8(nuovi parametri).pth" 
    
    GRID_SIZE = 8
    EPISODI_DI_TEST = 20

    try:
        test_model(model_path=NOME_FILE_MODELLO, grid_size=GRID_SIZE, num_episodes=EPISODI_DI_TEST)
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file '{NOME_FILE_MODELLO}'.")
        print("Assicurati di aver inserito il nome corretto generato alla fine di train.py.")