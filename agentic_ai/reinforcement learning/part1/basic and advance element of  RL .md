The framework of Reinforcement Learning (RL) is built upon **six core elements**. These elements work together in a continuous loop: the agent performs an action, the environment responds, and the agent learns from the feedback.

---

# The 6 Elements of Reinforcement Learning

## 1. The Agent
The **Agent** is the learner or the decision-maker. It is the "AI" or the entity that we are training. It has the power to observe the world and take actions to achieve a specific goal.
*   **Goal:** To maximize the total reward over time.
*   **Analogy:** A player in a video game.

## 2. The Environment
The **Environment** is everything outside of the agent. It is the world the agent lives in and interacts with. It receives an action from the agent, changes its state, and provides feedback (reward).
*   **Analogy:** The video game world itself (the levels, the physics, the enemies).

## 3. The State ($S$)
The **State** is a description of the current situation of the environment. It provides the information the agent needs to choose an action.
*   **Symbol:** $S_t$ (State at time $t$).
*   **Example:** In Chess, the position of every piece on the board is the state.

## 4. The Action ($A$)
The **Action** is what the agent decides to do in a given state. The set of all possible moves the agent can make is called the "Action Space."
*   **Symbol:** $A_t$ (Action taken at time $t$).
*   **Example:** Moving a chess piece to a specific square.

## 5. The Reward ($R$)
The **Reward** is the immediate feedback the environment sends back to the agent after an action. It tells the agent how "good" or "bad" the last action was.
*   **Symbol:** $R_t$ (Reward received at time $t$).
*   **Positive Reward:** Encourages the behavior (e.g., getting points).
*   **Negative Reward (Penalty):** Discourages the behavior (e.g., losing a life).

## 6. The Policy ($\pi$)
The **Policy** is the agent’s strategy or rulebook. It is the mapping from states to actions. It defines how the agent behaves in a particular situation.
*   **Symbol:** $\pi(a|s)$
*   **Types:** 
    *   **Deterministic:** Always takes action $A$ in state $S$.
    *   **Stochastic:** Has a probability of taking different actions in state $S$.

---

# Secondary (Advanced) Elements

Beyond the basics, RL systems often include these two components:

### 7. The Value Function ($V$)
While the Reward is *immediate*, the **Value Function** predicts the *long-term* future reward. It tells the agent if a state is "good" in the long run, even if the immediate reward is small.
*   **Logic:** A state might give a small reward now but lead to a massive jackpot later. The Value Function captures this potential.

### 8. The Model (Optional)
The **Model** is the agent's internal representation of the environment. It mimics the behavior of the real environment to help the agent plan ahead by predicting what the next state and reward will be before actually taking the action.
*   **Model-Based RL:** Uses a model to plan.
*   **Model-Free RL:** Learns purely by trial and error without a model.

---

# The RL Loop (How they connect)

1.  **Agent** observes the current **State** ($S_t$).
2.  **Agent** follows its **Policy** ($\pi$) to choose an **Action** ($A_t$).
3.  **Environment** reacts to the action.
4.  **Environment** transitions to a **New State** ($S_{t+1}$).
5.  **Environment** gives a **Reward** ($R_{t+1}$) to the agent.
6.  **Agent** updates its knowledge (**Value Function** or **Policy**) based on that reward.
7.  *Repeat until the goal is reached.*
