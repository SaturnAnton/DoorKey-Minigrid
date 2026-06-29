import torch
import numpy as np
import os

from env import MinigridDoorKeyFullyObs
from model import CnnMinigridPolicy

def grid_to_str(env):
    grid = env.unwrapped.grid
    width = env.unwrapped.width
    height = env.unwrapped.height
    agent_pos = env.unwrapped.agent_pos
    carrying = env.unwrapped.carrying

    has_key = carrying is not None and carrying.type == 'key'

    SYMBOLS = {
        'wall':   '▇',
        'door':   'D',
        'key':    'K',
        'goal':   'G',
        None:     ' ',
    }

    horizontal_border = '+' + '---+' * width
    rows = []

    for y in range(height):
        rows.append(horizontal_border)
        row_content = '|'
        for x in range(width):
            if (x, y) == tuple(agent_pos):
                row_content += f" { 'L' if has_key else 'A' } |"
            else:
                cell = grid.get(x, y)
                obj_type = cell.type if cell is not None else None
                
                if obj_type == 'door' and cell.is_open:
                    symbol = ' '  
                else:
                    symbol = SYMBOLS.get(obj_type, ' ')
                    
                row_content += f" {symbol} |"
        rows.append(row_content)
    
    rows.append(horizontal_border)
    return '\n'.join(rows)

def test_model(model_path, grid_size, num_episodes, output_log_path="sequenza_episodi.txt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing su dispositivo: {device}")

    env = MinigridDoorKeyFullyObs(size=grid_size, render=True)
    num_actions = env.action_space.n
    state_space = env.observation_space.shape

    print(f"Caricamento dei pesi dal file: {model_path}")
    model = CnnMinigridPolicy(input_shape=state_space, num_actions=num_actions).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_params'])
    
    model.eval()

    success_count = 0

    with open(output_log_path, "w", encoding="utf-8") as f_out:
        
        for episode in range(num_episodes):
            state = env.reset()[0]
            done = False
            step_count = 0
            episode_reward = 0

            print(f"--- Inizio Episodio {episode + 1} ---")
            f_out.write(f"=========================================\n")
            f_out.write(f"INIZIO EPISODIO {episode + 1}\n")
            f_out.write(f"=========================================\n\n")

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    q_values = model(state_tensor).cpu().numpy()
                
                action = np.argmax(q_values)

                state_str = grid_to_str(env)
                print(state_str)

                f_out.write(f"<step_{step_count}>\n")
                f_out.write(state_str + "\n")
                f_out.write(f"</step_{step_count}>\n")
                f_out.write("-" * 40 + "\n") 

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                state = next_state

            if episode_reward > 0:
                success_count += 1
                risultato_str = f"Risultato: VITTORIA in {step_count} passi! (Reward: {episode_reward:.3f})"
            else:
                risultato_str = f"Risultato: SCONFITTA/Timeout. (Passi: {step_count})"
            
            print(risultato_str)
            f_out.write(f"\n{risultato_str}\n\n")

        success_rate = (success_count / num_episodes) * 100
        riassunto = (
            f"\n=== Riassunto Test ===\n"
            f"Episodi giocati: {num_episodes}\n"
            f"Vittorie: {success_count}\n"
            f"Success Rate: {success_rate:.1f}%\n"
        )
        print(riassunto)
        f_out.write(riassunto)

    env.close()
    print(f"I dati degli step sono stati salvati con successo in: {output_log_path}")

if __name__ == "__main__":
    NOME_FILE_MODELLO = "data/22-8x8.pth" 
    GRID_SIZE = 8
    EPISODI_DI_TEST = 1
    FILE_OUTPUT_LLM = "sequenza_passi_llm.txt" # Nome del file di testo finale

    try:
        test_model(
            model_path=NOME_FILE_MODELLO, 
            grid_size=GRID_SIZE, 
            num_episodes=EPISODI_DI_TEST,
            output_log_path=FILE_OUTPUT_LLM
        )
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file '{NOME_FILE_MODELLO}'.")
        print("Assicurati di aver inserito il nome corretto generato alla fine di train.py.")