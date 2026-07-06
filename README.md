# Modelli visuali e linguistici per definire il comportamento di agenti intelligenti

## Indice
- [1. Reinforcement Learning](#1-reinforcement-learning)
  - [Conoscenza del Modello](#conoscenza-del-modello)
- [2. Ambiente Door Key](#2-ambiente-door-key)
  - [Spazio delle azioni](#spazio-delle-azioni)
- [3. Algoritmi: Q-Learning e SARSA (tabulare)](#3-algoritmi-q-learning-e-sarsa-tabulare)
  - [Q-Learning](#q-learning)
    - [Sample-based Q-Learning](#sample-based-q-learning)
    - [Proprietà e Convergenza](#proprietà-e-convergenza)
    - [Pseudocodice Q-Learning](#pseudocodice-q-learning)
  - [SARSA](#sarsa)
    - [Pseudocodice SARSA](#pseudocodice-sarsa)
    - [Convergenza](#convergenza)
  - [SARSA vs Q-Learning](#sarsa-vs-q-learning)
  - [Esplorazione vs Sfruttamento](#esplorazione-vs-sfruttamento)
  - [Parametri](#parametri)
  - [Setup dell'addestramento](#setup-delladdestramento)
  - [Training Q-Learning](#training-q-learning)
  - [Training SARSA](#training-sarsa)
- [4. Double DQN](#4-double-dqn)
  - [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Architettura della rete neurale](#architettura-della-rete-neurale)
    - [Blocco convoluzionale](#blocco-convoluzionale)
    - [Blocco fully-connected](#blocco-fully-connected-denso)
  - [Double DQN](#double-dqn)
- [5. Implementazione LLM e VLM](#5-implementazione-llm-e-vlm)
  - [Evoluzione dell'Architettura](#evoluzione-dellarchitettura)
  - [Analisi di un singolo episodio vincente](#analisi-di-un-singolo-episodio-vincente)

---

## 1. Reinforcement Learning
Il Reinforcement Learning (RL) è un paradigma dell'apprendimento automatico in cui un agente intelligente impara a prendere decisioni ottimali interagendo con un ambiente, con l'obiettivo di massimizzare la ricompensa.

I concetti fondamentali del RL includono:
 
- **Trial and error (Apprendimento per prove ed errori):** l'agente esplora l'ambiente in modo attivo, deducendo le strategie ottimali dalle conseguenze delle proprie azioni.
- **Delayed reward (Ricompensa ritardata):** le azioni intraprese possono avere conseguenze a lungo termine.

L'obiettivo formale del RL è avere la stima della **funzione di valore** `V(s)` a lungo termine e l'individuazione della **policy ottimale** `π(s)`che permette di massimizzare la ricompensa attesa.
 
Il Reinforcement Learning è formalizzabile come un **Markov Decision Process (MDP)** in cui, a differenza della pianificazione classica:
 
- La funzione di ricompensa `R(s, a, s')` è incognita.
- La probabilità di transizione tra stati `T(s, a, s')` sono incognite.

Di conseguenza, l'agente deve provare ciascuna azione e raccogliere le relative ricompense.
 
### Conoscenza del Modello
 
Le strategie per risolvere problemi di RL si dividono in due categorie principali:
* **Model-based**: cercano di apprendere un modello dell'ambiente, evitano di ripetere stati/azioni negative, richiedono meno passaggi di esecuzione e utilizzano i dati in modo efficiente.
* **Model-free**: cercano di apprendere direttamente la Q-function e la policy, si basano sulla semplicità cioè non è necessario cotruire e utilizzare un modello e presentano assenza di bias nella progettazione del modello.


<br><br>

---

## 2. Ambiente Door Key
Per valutare le performance dell'agente è stato scelto l'ambiente Door Key, della libreria MiniGrid. Questo ambiente presenta una chiave che l'agente deve raccogliere per sbloccare la porta e successivamente deve arrivare al quadrato verde che rappresenta il goal finale. Infatti la missione di questo ambiente è riassunta come: "usa la chiave per aprire la porta e poi raggiungi il goal". Poiché la ricompensa viene fornita esclusivamente al completamento del task finale (reward sparso), l'ambiente rappresenta un'ottima sfida per testare le capacità di esplorazione dell'algoritmo.

<div align="center">
  <img src="figure/env.png" alt="Ambiente di gioco" width="200">
  <p><i>Visualizzazione dell'ambiente Door Key in MiniGrid</i></p>
</div>

Il reward di questo ambiente può essere solo di due tipi:
- `1 − 0.9 · (step_count/max_steps)` se raggiunge il goal.
- `0` se fallisce.

L'episodio termina quando si verifica una delle seguenti condizioni:
- L'agente raggiunge con successo il goal.
- Viene superato il numero massimo di passi consentiti per un singolo episodio (definito dalla variabile `max_steps`).

### Spazio delle azioni
Le azioni possibili all'interno di questo ambiente sono 7, identificate da un numero:

| ID | Azione | Descrizione |
|----|--------|-------------|
| 0 | LEFT | Ruotare a sinistra |
| 1 | RIGHT | Ruotare a destra |
| 2 | FORWARD | Muoversi in avanti di una cella |
| 3 | PICKUP | Raccogliere un oggetto (la chiave) |
| 4 | DROP | Rilasciare un oggetto |
| 5 | TOGGLE | Interagire con un oggetto (aprire la porta) |
| 6 | DONE | Terminazione volontaria |

<br><br>

---

## 3. Algoritmi: Q-Learning e SARSA (tabulare)

### Q-Learning
 
Il Q-Learning è il principale algoritmo **model-free** progettato per permettere a un agente di apprendere come comportarsi in modo ottimale interagendo direttamente con l'ambiente. L'obiettivo di questo algoritmo è quello di stimare direttamente la Q-Function `Q(s,a)` senza conoscere in anticipo le dinamiche dell'ambiente.
 
$$Q_{k+1}(s,a) = \sum_{s'} T(s, a, s') \left( R(s, a, s') + \gamma \max_{a'} Q_k(s', a') \right)$$

Questa equazione permette di torvare iterativamente i valori di Q ottimi.
 
#### Sample-based Q-Learning
 
Acquisendo il campione `(s, a, s', r)`, l'algoritmo aggiorna la Q-Function facendo riferimento alla vecchia stima `Q(s,a)` prendendo in considerazione il nuovo campione:

$$sample = R(s, a, s') + \gamma \max_{a'} Q(s', a')$$

Si incorpora la nuova stima in una running average:
 
$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left( R(s, a, s') + \gamma \max_{a'} Q(s', a') \right)$$

La variabile α rappresenta il **learning rate**, cioè il peso che viene dato al nuovo
campione rispetto alla vecchia stima. Nel corso del tempo questo valore ciene decrementato per garantire la convergenza.
 
#### Proprietà e Convergenza
 
Il Q-Learning garantisce la convergenza alla policy ottima se si verificano queste due condizioni fondamentali:
 
- **Esplorazione adeguata:** Ogni coppia stato-azione deve essere visitata un numero infinito di volte nel lungo periodo.
- **Decadimento di α:** il learning rate deve diminuire progressivamente (es. `α = 1 / n(s,a)`).

Il Q-Learning è un algoritmo **Off-policy**: separa la policy usata per generare il comportamento (es. esplorazione epsilon-greedy) dalla policy che viene valutata e ottimizzata (puramente greedy, rappresentata dall'operatore `max_a'`).
 
#### Pseudocodice Q-Learning
 
```
Initialize Q(s, a) arbitrarily for all s in S, a in A(s)
Initialize Q(terminal-state, .) = 0
 
Repeat (for each episode):
  Initialize s
  Repeat (for each step of episode):
    Choose a from s using policy derived from Q (e.g., epsilon-greedy)
    Take action a, observe r, s'
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    s <- s'
  until s is terminal
```

---

### SARSA

L'algoritmo SARSA prende il nome dalla tupla da cui deriva: `(S, A, R, S', A')`, ovvero stato, azione, ricompensa, stato successivo e azione successiva.

La caratteristica principale di questo algoritmo è che il calcolo dell'azione successiva avviene seguendo la stessa politica che l'agente sta utilizzando per agire nell'ambiente: per questo motivo SARSA è definito un algoritmo on-policy.

#### Pseudocodice SARSA

```
Initialize Q(s, a) arbitrarily for all s in S, a in A(s)
Initialize Q(terminal-state, .) = 0

Repeat (for each episode):
  Initialize S
  Choose A from S using policy derived from Q (e.g., ε-greedy)
  Repeat (for each step of episode):
    Take action A, observe R, S'
    Choose A' from S' using policy derived from Q (e.g., ε-greedy)
    Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
    S <- S'; A <- A'
  until S is terminal
```
#### Convergenza
Se la politica converge, nel limite, verso la politica greedy (cioè quella che sceglie sempre l'azione con il valore massimo), e a condizione che ogni coppia stato-azione venga visitata infinite volte, allora SARSA garantisce la convergenza verso la funzione Q ottima, `Q*(s, a)`.

### SARSA vs Q-Learning

| | Q-Learning | SARSA |
|---|---|---|
| **Tipo** | Off-policy | On-policy |
| **Aggiornamento** | Usa `max_a' Q(S', a')` | Usa `Q(S', A')` con A' dalla policy |
| **Policy ottima** | Sì, la impara direttamente | Sì, ma solo se la policy converge a greedy |
| **Performance online** | Peggiore | Migliore |
| **Rischio** | Può fallire occasionalmente con ε-greedy | Più cauto, evita azioni rischiose |

Il classico esempio del **Cliff Walking** mostra questa differenza: Q-Learning impara il percorso ottimo teorico ma ci cade occasionalmente a causa dell'esplorazione ε-greedy, mentre SARSA apprende un percorso leggermente più lungo ma più sicuro, ottenendo una ricompensa cumulativa migliore durante il training.
 
### Esplorazione vs Sfruttamento
 
Un pilastro fondamentale del Reinforcement Learning è il delicato compromesso tra **esplorazione** (exploration) e **sfruttamento** (exploitation):
 
- **Esplorazione:** l'agente sacrifica il guadagno immediato per raccogliere informazioni che potrebbero portare a profitti molto più alti in futuro.
- **Sfruttamento:** l'agente sceglie le azioni che hanno dato buoni risultati in passato per ottenere ricompense immediate e sicure.

Per garantire la convergenza a un risultato ottimo, è necessario esplorare tutte le coppie stato-azione con sufficiente frequenza nel lungo periodo. I principali metodi utilizzati nella pratica sono:
- **ε-greedy**: Scegliere l'azione in modo greedy per la maggior parte del tempo (con probabilità `1 - ε`) e selezionare un'azione casuale con probabilità `ε`. Questo metodo è off-policy.
- **Softmax**: Viene scelta una azione `a` con una probabilità:

  $$p(a) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}$$

  Il parametro **T** (temperatura) agisce da modulatore del grado di esplorazione:

  | Temperatura | Comportamento |
  |-------------|---------------|
  | T elevata   | Tutte le azioni hanno probabilità simili → **massima esplorazione** |
  | T bassa     | L'azione con Q più alto ha probabilità maggiore → **massimo sfruttamento** |
 
### Parametri
 
| Parametro | Simbolo | Descrizione |
|-----------|---------|-------------|
| Episodi | — | Sequenze complete di interazioni agente-ambiente, dall'inizio fino allo stato terminale o al limite massimo di passi (`max_steps`). |
| Learning Rate | α | Determina il peso delle nuove informazioni rispetto a quelle passate. Regola la velocità di aggiornamento dei valori nella Q-table. |
| Discount Factor | γ | Definisce l'importance delle ricompense future; un valore vicino a 1 orienta l'agente verso una strategia di lungo periodo. |
| Exploration Rate | ε | Probabilità di scegliere un'azione casuale invece di quella ottimale stimata; fondamentale per esplorare l'ambiente. |
| Temperatura | T | Regola il grado di esplorazione nella strategia softmax, bilanciando l'estrazione casuale di azioni con la preferenza per quelle con valori Q più alti. |

### Setup dell'addestramento
La fase di addestramento è stata strutturata in due diverse configurazioni sperimentali per valutare l’impatto dei parametri esplorativi e della durata massima degli episodi sulle performance di convergenza dell’agente. 
Entrambi i set di esperimenti sono stati condotti per un totale di 5000 episodi, mantenendo costanti il learning rate (α) e il fattore di sconto (γ). I test sono stati suddivisi come segue:
- Configurazione 1: Parametri impostati con α = 0.1, γ = 0.99 ed un tasso di esplorazione iniziale ε oppure TEMP = 0.9. Il limite massimo di passi per episodio (max_step) è stato mantenuto al valore di default dell’ambiente, pari a 250.
- Configurazione 2: Parametri impostati con α = 0.1, γ = 0.99. In questo caso, si è optato per una strategia di esplorazione iniziale più aggressiva, impostando ε oppure TEMP = 1.0. Inoltre, per concedere all’agente un orizzonte temporale più ampio per l’esplorazione casuale, il limite di passi è stato raddoppiato, portando il max_step a 500.

### Training Q-Learning
È stato svolto il training dell'agente con le due configurazioni diverse per entrambi i metodi di esplorazione:
- ε-greedy
  * Configurazione 1: dalla 1 alla 5
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable1.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable2.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable3.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable4.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable5.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
  * Configurazione 2: dalla 6 alla 10
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable6.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable7.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable8.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable9.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable10.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
- Softmax
  * Configurazione 2: dalla 11 alla 15
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable11.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable12.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable13.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable14.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable15.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
  * Configurazione 1: dalla 16 alla 20
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable16.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable17.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable18.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/Q-Learning/qtable19.png" width="100%"/></td>
        <td align="center"><img src="figure/Q-Learning/qtable20.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>

I risultati dei test sono disponibili nella cartella [Q-Learning](./figure/Q-Learning/).

### Training SARSA
È stato svolto il training dell'agente con le due configurazioni diverse per entrambi i metodi di esplorazione:
- ε-greedy
  * Configurazione 1: dalla 1 alla 5
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa1.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa2.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa3.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa4.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa5.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
  * Configurazione 2: dalla 6 alla 10
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa6.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa7.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa8.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa9.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa10.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
- Softmax
  * Configurazione 2: dalla 11 alla 15
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa11.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa12.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa13.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa14.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa15.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>
  * Configurazione 1: dalla 16 alla 20
    <table align="center" width="100%">
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa16.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa17.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa18.png" width="100%"/></td>
      </tr>
      <tr>
        <td align="center"><img src="figure/SARSA/sarsa19.png" width="100%"/></td>
        <td align="center"><img src="figure/SARSA/sarsa20.png" width="100%"/></td>
        <td></td>
      </tr>
    </table>

I risultati dei test sono disponibili nella cartella [SARSA](./figure/SARSA/).

<br><br>

---

## 4. Double DQN
### Deep Q-Network (DQN)
Il Q-Learning tabulare mantiene una tabella esplicita dei valori Q per ogni coppia stato-azione, il che lo rende inapplicabile in contesti con spazi di azione continui o ad alta dimensionalità, dove il numero di combinazioni possibili diventa intrattabile. Per superare questo limite, si ricorre al [Deep Q-Network (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), che sostituisce la Q-table con una rete neurale in grado di approssimare la funzione Q su spazi continui, rendendo possibile l'apprendimento in ambienti molto più complessi. Lo pseudocodice del DQN è il seguente:

```
Initialize weights w and w ’ randomly in [ -1 , 1]
Initialize s { observe current state }
loop
  Select and execute action a
  Observe new state s ’ receive immediate reward r
  Add (s , a , s ’, r ) to experience buffer
  Sample mini - batch MB of experiences from buffer
  for ( s_hat , a_hat , s ’, r_hat ) in MiniBatch do
    grad = ( Q_w ( s_hat , a_hat ) - r_hat - gamma * max_a_hat ’( Q_w_hat (
    s_hat ’ , a_hat ’) )) * partial Q_w ( s_hat , a_hat )/ partial w
    update weights w <- w - alpha * grad
  end for
  update state s <- s ’
  every c steps , update target : w <- w
end loop
```

### Architettura della rete neurale

Come rete neurale è stata implementata una **rete neurale convoluzionale (CNN)**. Una CNN è un tipo di rete neurale particolarmente adatta all'elaborazione di dati con struttura a griglia, come le immagini: invece di collegare ogni neurone a tutti i pixel in ingresso (come avviene nelle reti completamente connesse), utilizza dei filtri (kernel) che scorrono sull'immagine per estrarre caratteristiche locali come bordi, forme e pattern. Questo permette di catturare le relazioni spaziali tra i pixel riducendo drasticamente il numero di parametri da addestrare rispetto a una rete densa equivalente. L'architettura di questa rete è formata da un blocco convoluzionale e un blocco fully-connected.

#### Blocco convoluzionale

Composto da due layer `Conv2d`, ognuno seguito da un'attivazione `ReLU`:

- Il **primo layer** prende in ingresso l'immagine con `C` canali (a seconda della rappresentazione dell'ambiente) e produce **32 mappe di caratteristiche** (feature map), applicando filtri di dimensione 3x3 con padding 1 (in modo da mantenere invariate altezza e larghezza dell'immagine).
- Il **secondo layer** prende le 32 mappe e ne produce **64**, sempre con un filtro 3x3 e padding 1.

Lo scopo di questi layer è estrarre progressivamente caratteristiche visive sempre più astratte dall'immagine dell'ambiente di gioco.

#### Blocco fully-connected (denso)

Dopo i layer convoluzionali, l'output viene appiattito (`flatten`) in un unico vettore e passato attraverso tre layer lineari:

- Il **primo** riduce il vettore a **256 neuroni**, seguito da una `ReLU`.
- Il **secondo** lo riduce ulteriormente a **64 neuroni**, anch'esso seguito da una `ReLU`.
- Il **terzo e ultimo layer** produce in uscita `num_actions` valori (uno per ogni azione possibile dell'agente), senza attivazione, poiché questi valori rappresentano direttamente i **Q-value** stimati per ciascuna azione.

### Double DQN
**Double DQN (DDQN)** è una variante migliorata del classico DQN. Il DQN standard tende a sovrastimare i Q-value durante il training, poiché utilizza la stessa rete sia per scegliere l'azione migliore nello stato successivo, sia per valutarne il Q-value. Questo fenomeno, noto come overestimation bias, porta a stime instabili e a un apprendimento più lento.
 
Il Double DQN risolve questo problema separando i due ruoli tra due reti distinte:
 
- La rete **online** (`dqn`) sceglie quale sia l'azione migliore nel prossimo stato.
- La rete **target** (`dqn_target`) valuta il Q-value di quella specifica azione.

Nel codice, questa logica è implementata nel ciclo di training in `train.py`:
 
```python
q_next_online = dqn(next_states_t)                    # rete online: sceglie l'azione
best_actions = torch.argmax(q_next_online, dim=1)
q_next_target = dqn_target(next_states_t)             # rete target: valuta l'azione
q_next_value = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
targets = reward_mb + gamma * q_next_value * (1 - done_mb)
```
 
La rete target viene sincronizzata con la rete online ogni `target_update_freq` step tramite un **hard update**, ovvero copiando direttamente i pesi:
 
```python
def hard_update(local_model, target_model):
    target_model.load_state_dict(local_model.state_dict())
```
 
Questo disaccoppiamento riduce la sovrastima dei Q-value e garantisce un addestramento più stabile e affidabile.

<br><br>

---


## 5. Implementazione LLM e VLM

Per superare il problema del reward sparso (nello scenario di default il reward veniva assegnato solo al raggiungimento del goal o al superamento del limite massimo di passi), si è deciso di integrare modelli linguistici e di visione (LLM e VLM) all'interno del loop di training della Double DQN. L'obiettivo è generare un reward denso a ogni singolo step di ciascun episodio, guidando l'agente in modo più efficace.

Il processo di sviluppo ha seguito un'evoluzione iterativa per ottimizzare tempi di risposta e vincoli di computazione:

### Evoluzione dell'Architettura

* **1. VLM Locale (Approccio Iniziale)**
    * Tentativo: Installazione di una VLM locale per testarne le capacità di analisi visiva dell'ambiente.
    * Criticità: Tempi di inferenza eccessivamente lunghi, anche dopo aver effettuato il downgrade a un modello più leggero.
* **2. VLM Cloud via Groq**
    * Tentativo: Migrazione a un provider esterno (Groq) per sfruttare l'accelerazione hardware e ridurre la latenza.
    * Criticità: Sebbene il tempo di risposta fosse migliorato, i limiti di Token Per Minute (TPM) e Request Per Minute (RPM) del piano gratuito hanno rallentato drasticamente il training globale.
* **3. Transizione a LLM Cloud via Cerebras**
    * Tentativo: Switch da VLM (Vision) a LLM (Text-only) per ridurre il consumo di token, permettendo l'esecuzione di un numero maggiore di episodi.
    * Criticità: La dipendenza da API esterne esponeva comunque il sistema a limitazioni tariffarie giornaliere e orarie (Rate Limiting).
* **4. LLM Locale e l'Ottimizzazione con Ollama**
    * Soluzione Finale: Configurazione di un modello linguistico locale per eliminare la dipendenza dai provider esterni e sbloccare un training continuo senza limiti di richieste.
    * Risultato: L'integrazione finale tramite Ollama ha abbattuto i tempi di risposta da ~2.0 secondi a soli 0.2 secondi per step. Questo incremento prestazionale ha permesso di scalare notevolmente il numero di episodi completati nell'unità di tempo.
 
### Analisi di un singolo episodio vincente

Nonostante il tempo di risposta molto basso per ogni step, eseguire un training completo di 7000 episodi da 1000 step ciascuno avrebbe richiesto tempi eccessivamente lunghi. Inoltre, nel corso degli addestramenti eseguiti con un numero limitato di episodi (da 30 a 300), si è osservato che il prompt, la rappresentazione dell'ambiente tramite stringhe e il modello utilizzato non permettevano di ottenere risultati corretti per quanto riguarda il reward relativo a ogni singolo step (reward denso).

Per questo motivo si è scelto di utilizzare i risultati del training della rete neurale, salvando su un file esterno l'immagine dell'ambiente per ogni step, e di applicare successivamente l'LLM a un numero ristretto di step, selezionati tra quelli che conducevano al goal. Fatto ciò, sono stati modificati i tre aspetti indicati in precedenza:

- **Modello**: è stato impiegato il modello più potente a disposizione (`gpt-oss-120b`), al fine di garantire una maggiore affidabilità nei risultati ottenuti.
- **Rappresentazione dell'ambiente**: l'ambiente viene rappresentato tramite caratteri specifici, secondo la seguente legenda:
  - `A` = agente senza la chiave (Agent without the key)
  - `L` = agente con la chiave (Loaded)
  - `K` = la chiave (Key)
  - `D` = la porta (Door)
  - `G` = il goal (Goal)
  - `▇` = muro, non attraversabile (Wall)
  - ` ` = spazio vuoto (Empty space)
- **Prompt**: è stato adattato sulla base delle modifiche apportate alla rappresentazione dell'ambiente.

Grazie a queste modifiche, si è ottenuto il valore esatto del reward denso per ogni singolo step già alla prima esecuzione. Proseguendo con le simulazioni, tuttavia, è emerso che i risultati non erano sempre corretti. Si è quindi deciso di utilizzare lo stesso prompt e la stessa rappresentazione con modelli diversi: a differenza dei modelli `gpt-oss-120b` e `qwen3-32b`, che sono riusciti almeno una volta a ottenere tutti gli step corretti, gli altri modelli hanno presentato, in ogni episodio, almeno uno step con un risultato errato.

| Modello       | Step corretti | Episodi corretti | % successo step | % successo episodi |
|----------------|:---------------:|:-------------------:|:------------------:|:----------------------:|
| gpt-oss-120b      | 42/52            | 1/4                 | 81%                 | 25%                     |
| llama-3.3-70b-versatile      | 0/52            | 0/4                 | 0%                 | 0%                     |
| qwen3-32b      | 46/52            | 2/4                 | 88%                 | 50%                     |
| qwen3.6-27b      | 35/52            | 0/4                 | 67%                 | 0%                     |
| zai-glm-4.7     | 30/52            | 0/4                 | 58%                 | 0%                     |

Per questo motivo, il prompt è stato ulteriormente modificato, specificando in maniera più dettagliata la situazione che generava il maggior numero di problemi, ossia il momento in cui l'agente apriva la porta. Una volta implementato il nuovo prompt, sono stati eseguiti nuovamente i test su tutti i modelli analizzati in precedenza: il modello `gpt-oss-120b` ha ottenuto sempre risultati corretti per ogni step di ogni episodio, mentre tutti gli altri modelli, così come con il prompt precedente, hanno continuato a presentare almeno un risultato errato nel reward per almeno uno step per episodio.

| Modello       | Step corretti | Episodi corretti | % successo step | % successo episodi |
|----------------|:---------------:|:-------------------:|:------------------:|:----------------------:|
| gpt-oss-120b      | 52/52            | 4/4                 | 100%                 | 100%                     |
| llama-3.3-70b-versatile      | 0/52            | 0/4                 | 0%                 | 0%                     |
| qwen3-32b      | 50/52            | 2/4                 | 96%                 | 50%                     |
| qwen3.6-27b      | 43/52            | 0/4                 | 82%                 | 0%                     |
| zai-glm-4.7     | 42/52            | 0/4                 | 81%                 | 0%                     |

Dai test condotti si può quindi concludere che questo tipo di task può essere risolto efficacemente solo da un modello particolarmente potente: modelli con un numero inferiore di parametri tendono infatti a commettere errori in almeno uno step.
