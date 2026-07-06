import os
import re
import time
from monorepo import GroqLLM, load_api_keys

def reward_llm(state, client, prompt, max_retries=8):
    full_prompt = f"{prompt}\n\nCURRENT STATE:\n{state}"
    
    time.sleep(5)

    for attempt in range(max_retries):
        try:
            risposta = client.ask(prompt=full_prompt)
            raw_content = risposta.strip()
            numbers = re.findall(r'-?\.?\d+\.?\d*', raw_content)

            if numbers:
                ultimo_numero = numbers[-1]
                try:
                    val = float(ultimo_numero)
                    if val > 0 or val == -0.005:
                        return val
                    else:
                        print(f"   [Parsing] Valore non conforme: '{val}' → reward = -0.005")
                        return -0.005
                except ValueError:
                    pass

            print(f"   [Parsing] Nessun numero valido trovato nell'output del LLM → reward = -0.005")
            return -0.005

        except Exception as e:
            if "429" in str(e) or "queue_exceeded" in str(e):
                wait = (2 ** attempt) + 60
                print(f"   [Rate limit Groq] tentativo {attempt+1}/{max_retries}: {e} — attendo {wait:.1f}s")
            else:
                wait = min(2 ** attempt, 5)
                print(f"   [Errore VLM] tentativo {attempt+1}/{max_retries}: {e} — attendo {wait}s")
            
            if attempt == max_retries - 1:
                print("   [Max retries] Groq non disponibile dopo i tentativi → reward = -0.005")
                return -0.005
            
            time.sleep(wait)

    return -0.005

def analizza_log_episodi(prompt_path, log_path):
    load_api_keys()
    client = GroqLLM(model_id= "qwen/qwen3.6-27b")

    if not os.path.exists(prompt_path):
        print(f"ERRORE: Il file di prompt '{prompt_path}' non esiste.")
        return
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_istruzioni = f.read()

    if not os.path.exists(log_path):
        print(f"ERRORE: Il file di log '{log_path}' non esiste.")
        return
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    print("=" * 60)
    print(f"AVVIO VALUTAZIONE LOG CON LLM LOCALE")
    print("=" * 60)

    step_pattern = re.compile(r"<step_(\d+)>(.*?)</step_(\d+)>", re.DOTALL)
    steps_trovati = step_pattern.findall(log_content)

    if not steps_trovati:
        print("Nessuno step valido trovato nel file di log. Controlla i tag <step_X>.")
        return

    for step_info in steps_trovati:
        step_num = step_info[0]      
        grid_state = step_info[1].strip() 

        print(f"\n[ANALISI] Inviando Step {step_num} all'LLM...")
        
        dense_reward = reward_llm(grid_state, client, prompt_istruzioni)

        print(f"--- Risultato Elaborazione Step {step_num} ---")
        print(grid_state)
        print(f"-> OLLAMA DENSE REWARD: {dense_reward}")
        print("-" * 40)

    print("\n[FINE] Elaborazione di tutti gli step completata.")

if __name__ == "__main__":
    PROMPT_FILE = "prompt3.txt"
    LOG_FILE = "sequenza_passi_llm.txt"

    analizza_log_episodi(prompt_path=PROMPT_FILE, log_path=LOG_FILE)