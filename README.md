# Modelli visuali e linguistici per definire il comportamento di agenti intelligenti

## Indice
- [1. Reinforcement Learning](#1-reinforcement-learning)
  - [Conoscenza del Modello](#conoscenza-del-modello)
- [2. Ambiente Doorkey](#2-ambiente-doorkey)
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
    - [ε-greedy](#ε-greedy)
    - [Softmax (Boltzmann)](#softmax-boltzmann)
  - [Parametri](#parametri)
  - [Setup dell'addestramento](#setup-delladdestramento)
  - [Training Q-Learning](#training-q-learning)
  - [Training SARSA](#training-sarsa)
- [4. Double DQN](#4-double-dqn)
  - [Architettura della rete neurale](#architettura-della-rete-neurale)
    - [Blocco convoluzionale](#blocco-convoluzionale)
    - [Blocco fully-connected](#blocco-fully-connected-denso)
- [5. Implementazione LLM e VLM](#5-implementazione-llm-e-vlm)
  - [Evoluzione dell'Architettura](#evoluzione-dellarchitettura)

---

## 1. Reinforcement Learning
Il Reinforcement Learning (RL) è un paradigma dell'apprendimento automatico in cui un agente intelligente impara a prendere decisioni ottimali interagendo con un ambiente, con l'obiettivo di massimizzare un segnale di ricompensa cumulativo nel tempo.
 
I concetti fondamentali del RL includono:
 
- **Trial and error (Apprendimento per prove ed errori):** l'agente esplora l'ambiente in modo attivo, deducendo le strategie ottimali dalle conseguenze delle proprie azioni.
- **Delayed reward (Ricompensa ritardata):** le azioni intraprese possono avere conseguenze e generare ricompense solo a lungo termine.

L'obiettivo formale del RL è la stima accurata della **funzione di valore** `V(s)` e l'individuazione della **policy ottima** `π*(s)`, ovvero la mappatura stato-azione che massimizza il valore atteso della ricompensa futura.
 
Il problema è formalizzabile como un **Processo Decisionale di Markov (MDP)** in cui, a differenza della pianificazione classica:
 
- La funzione di ricompensa `R(s, a, s')` è incognita.
- La probabilità di transizione tra stati `T(s, a, s')` (la dinamica dell'ambiente) sono incognite.

Di conseguenza, l'agente deve acquisire campioni empirici provando azioni e raccogliendo le relative ricompense.
 
### Conoscenza del Modello
 
Le strategie per risolvere problemi di RL si dividono in due categorie principali:
 
| | Model-based | Model-free |
|---|---|---|
| **Approccio** | Stima esplicitamente il modello dell'ambiente (matrici di transizione e di ricompensa) | Apprende direttamente la Q-function e la policy ottima tramite l'esperienza |
| **Vantaggi** | Maggiore efficienza di campionamento (sample efficiency); può pianificare riducendo l'interazione con stati negativi | Più semplice da implementare; bias inferiore |
| **Svantaggi** | Computazionalmente più complesso | Richiede una maggiore quantità di dati per convergere |

<br><br>

---

## 2. Ambiente Doorkey
Per valutare le performance dell'agente è stato scelto l'ambiente DoorKey, della libreria MiniGrid. Questo ambiente presenta una chiave che l'agente deve raccogliere per sbloccare la porta e successivamente deve arrivare al quadrato verde che rappresenta il goal finale. Infatti la missione di questo ambiente è riassunta come: "usa la chiave per aprire la porta e poi raggiungi il goal". Poiché la ricompensa viene fornita esclusivamente al completamento del task finale (ricompensa sparsa), l'ambiente rappresenta un'ottima sfida per testare le capacità di esplorazione dell'algoritmo.

<div align="center">
  <img src="figure/env.png" alt="Ambiente di gioco" width="200">
  <p><i>Visualizzazione dell'ambiente DoorKey in MiniGrid</i></p>
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
 
Il Q-Learning è il principale algoritmo **model-free** per il controllo ottimo. Permette di stimare direttamente i valori di `Q(s,a)` approssimando l'equazione di ottimalità di Bellman senza necessitare della dinamica dell'ambiente:
 
$$Q_{k+1}(s,a) = \sum_{s'} T(s, a, s') \left( R(s, a, s') + \gamma \max_{a'} Q_k(s', a') \right)$$
 
### Sample-based Q-Learning
 
Acquisendo una tupla di esperienza `(s, a, r, s')`, l'algoritmo aggiorna iterativamente la stima della funzione Q fondendo il valore storico con il nuovo target temporale (TD Target):
 
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right)$$
 
### Proprietà e Convergenza
 
Il Q-Learning garantisce la convergenza asintotica alla reale funzione `Q*(s,a)` (e quindi alla policy ottima) sotto due condizioni fondamentali:
 
- **Esplorazione adeguata:** ogni coppia stato-azione `(s,a)` deve essere visitata un numero infinitamente grande di volte.
- **Decadimento di α:** il learning rate deve diminuire progressivamente (es. `α = 1 / n(s,a)`).

Il Q-Learning è intrinsecamente un algoritmo **Off-policy**: separa la policy usata per generare il comportamento (es. esplorazione epsilon-greedy) dalla policy che viene valutata e ottimizzata (puramente greedy, rappresentata dall'operatore `max_a'`).
 
### Pseudocodice Q-Learning
 
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

SARSA è un algoritmo **on-policy** alternativo al Q-Learning per il model-free RL. Il nome deriva dalla tupla di esperienza che utilizza per l'aggiornamento: **(S, A, R, S', A')**.

La regola di aggiornamento è:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$$

La differenza chiave rispetto al Q-Learning è che **la prossima azione A' viene scelta seguendo la policy corrente** (es. ε-greedy), non prendendo il massimo. Per questo è detto on-policy.

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

##### Convergenza

SARSA converge alla policy ottima `Q*(s, a)` se, nel limite, la policy converge alla policy greedy **e** tutte le coppie stato-azione vengono visitate infinite volte.

##### SARSA vs Q-Learning

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
 
#### ε-greedy
 
Scegliere l'azione in modo greedy per la maggior parte del tempo (con probabilità `1 - ε`) e selezionare un'azione casuale con probabilità `ε`.
 
#### Softmax (Boltzmann)
 
Assegna una probabilità di selezione a ciascuna azione ponderandola in base al suo valore atteso `Q(s,a)`. La probabilità di scegliere l'azione `a` nello stato `s` è:
 
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
