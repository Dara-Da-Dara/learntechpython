In Reinforcement Learning, specific mathematical symbols are used to describe the interaction between the Agent and the Environment.

Here is a comprehensive guide to the symbols and their representations in **Markdown** format.

---

# Symbols and Representations in Reinforcement Learning

## 1. Core Variables (The RL Loop)
These symbols represent the data exchanged at every time step $t$.

| Symbol | Name | Description |
| :--- | :--- | :--- |
| $t$ | **Time step** | The current moment in time (0, 1, 2, ...). |
| $S_t$ | **State** | The current situation/snapshot of the environment at time $t$. |
| $A_t$ | **Action** | The decision or move made by the agent at time $t$. |
| $R_t$ | **Reward** | The numerical feedback received after taking action $A_{t-1}$. |
| $S_{t+1}$ | **Next State** | The state the agent lands in after taking an action (also written as $s'$). |

---

## 2. Sets (Spaces)
These represent the "collection" of all possible things the agent can see or do.

| Symbol | Name | Description |
| :--- | :--- | :--- |
| $\mathcal{S}$ | **State Space** | The set of all possible states in the environment. |
| $\mathcal{A}$ | **Action Space** | The set of all possible actions the agent can take. |
| $\mathcal{R}$ | **Reward Space** | The set of all possible reward values. |

---

## 3. Functions and Policies
These are the "brains" and "calculators" of the RL agent.

| Symbol | Name | Description |
| :--- | :--- | :--- |
| $\pi$ | **Policy** | The strategy: a mapping from states to actions. |
| $\pi(a\|s)$ | **Stochastic Policy** | The probability of taking action $a$ given state $s$. |
| $V(s)$ | **State-Value Function** | The expected long-term return starting from state $s$. |
| $Q(s, a)$ | **Action-Value Function** | The expected return starting from state $s$, taking action $a$. |
| $G_t$ | **Return** | The total accumulated reward from time $t$ until the end. |

---

## 4. Hyperparameters (The Greek Symbols)
These are settings defined by the programmer to control how the agent learns.

| Symbol | Name | Description |
| :--- | :--- | :--- |
| $\gamma$ | **Gamma** | **Discount Factor** (0 to 1): Determines the importance of future rewards. |
| $\alpha$ | **Alpha** | **Learning Rate**: How much new information overrides old information. |
| $\epsilon$ | **Epsilon** | **Exploration Rate**: The probability of picking a random action. |
| $\tau$ | **Tau** | **Temperature**: Used in Softmax to control "randomness" in choice. |

---

## 5. Mathematical Representations

### The Return Formula ($G_t$)
The total reward the agent expects to get from now until the future:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### The Policy ($\pi$)
A stochastic policy (probability distribution):
$$\pi(a|s) = P(A_t = a | S_t = s)$$

### Bellman Expectation Equation
How we represent the relationship between the current state and the next state:
$$V(s) = \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) | S_t = s]$$

---

## 6. The "S-A-R-S-A" Cycle
In many RL papers, you will see a trajectory represented like this:

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots$$

**This represents:**
1. Start in **State 0**.
2. Take **Action 0**.
3. Receive **Reward 1**.
4. Move to **State 1**.
5. *Repeat.*
