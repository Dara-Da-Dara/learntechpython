# Reinforcement Learning (RL) Vocabulary

This document presents **core Reinforcement Learning terminology** with:
- **Standard symbols**
- **Clear meaning**
- **Why the term is important (relevance)**
- **Real-world examples**
- **Scope in academic and industry projects**


---

## 1. Agent (𝒜)

**Meaning:**  
The learner or decision-maker that interacts with the environment.

**Symbol:**  
𝒜

**Use / Relevance:**  
Central entity in RL; responsible for learning optimal behavior.

**Real-World Example:**  
A robot navigating a warehouse.

**Scope in Projects:**  
Robotics, autonomous vehicles, recommender systems.

---

## 2. Environment (𝓔)

**Meaning:**  
Everything external to the agent with which it interacts.

**Symbol:**  
𝓔

**Use / Relevance:**  
Defines dynamics, constraints, and feedback.

**Real-World Example:**  
Road traffic system for a self-driving car.

**Scope in Projects:**  
Simulators, digital twins, smart cities.

---

## 3. State (s ∈ S)

**Meaning:**  
Representation of the current situation of the agent.

**Symbol:**  
s ∈ S

**Use / Relevance:**  
Basis for decision-making.

**Real-World Example:**  
Robot position and battery level.

**Scope in Projects:**  
State design is crucial in robotics and finance projects.

---

## 4. Action (a ∈ A)

**Meaning:**  
A choice made by the agent that affects the environment.

**Symbol:**  
a ∈ A

**Use / Relevance:**  
Defines agent’s control capability.

**Real-World Example:**  
Accelerate, brake, or turn steering wheel.

**Scope in Projects:**  
Control systems, robotics, games.

---

## 5. Reward (r or R)

**Meaning:**  
Scalar feedback signal indicating desirability of an action.

**Symbol:**  
r, R(s,a)

**Use / Relevance:**  
Drives learning and optimization.

**Real-World Example:**  
Positive reward for reaching destination safely.

**Scope in Projects:**  
Critical in healthcare, finance, and operations research.

---

## 6. Policy (π)

**Meaning:**  
Mapping from states to actions.

**Symbol:**  
π(a|s)

**Use / Relevance:**  
Defines agent behavior.

**Real-World Example:**  
Driving rules learned by an autonomous car.

**Scope in Projects:**  
Robotics control, resource allocation.

---

## 7. Value Function (V, Q)

**Meaning:**  
Expected cumulative reward from a state or state-action pair.

**Symbols:**  
V(s), Q(s,a)

**Use / Relevance:**  
Evaluates long-term benefit.

**Real-World Example:**  
Expected profit from a trading position.

**Scope in Projects:**  
Finance, recommendation systems.

---

## 8. Discount Factor (γ)

**Meaning:**  
Controls importance of future rewards.

**Symbol:**  
γ ∈ [0,1]

**Use / Relevance:**  
Balances short-term vs long-term goals.

**Real-World Example:**  
Immediate fuel savings vs long-term maintenance.

**Scope in Projects:**  
Economics, sustainability planning.

---

## 9. Exploration (ε)

**Meaning:**  
Trying new actions to gain knowledge.

**Symbol:**  
ε (epsilon)

**Use / Relevance:**  
Prevents premature convergence.

**Real-World Example:**  
Trying a new route to avoid traffic.

**Scope in Projects:**  
Recommendation systems, A/B testing.

---

## 10. Exploitation

**Meaning:**  
Using known best actions.

**Symbol:**  
—

**Use / Relevance:**  
Maximizes accumulated reward.

**Real-World Example:**  
Choosing a proven profitable strategy.

**Scope in Projects:**  
Business optimization, marketing.

---

## 11. Model (P, R)

**Meaning:**  
Representation of environment dynamics.

**Symbols:**  
P(s'|s,a), R(s,a)

**Use / Relevance:**  
Enables planning.

**Real-World Example:**  
Traffic flow simulation.

**Scope in Projects:**  
Digital twins, industrial automation.

---

## 12. Episode

**Meaning:**  
One complete sequence of interaction.

**Symbol:**  
—

**Use / Relevance:**  
Defines learning horizon.

**Real-World Example:**  
One full game of chess.

**Scope in Projects:**  
Games, simulations.

---

## 13. Temporal Difference Error (δ)

**Meaning:**  
Difference between predicted and actual reward.

**Symbol:**  
δ

**Use / Relevance:**  
Core learning signal.

**Real-World Example:**  
Prediction error in stock price movement.

**Scope in Projects:**  
Forecasting, adaptive control.

---

## 14. Experience Replay (𝒟)

**Meaning:**  
Memory of past transitions.

**Symbol:**  
𝒟 = (s,a,r,s')

**Use / Relevance:**  
Improves sample efficiency.

**Real-World Example:**  
Learning from recorded driving data.

**Scope in Projects:**  
Deep RL, autonomous systems.

---

## 15. Terminal State (s_T)

**Meaning:**  
End of an episode.

**Symbol:**  
s_T

**Use / Relevance:**  
Defines completion.

**Real-World Example:**  
Goal reached or failure in a task.

**Scope in Projects:**  
Robotics, task automation.

---

## Final Note on Scope

Mastery of this vocabulary is essential for:
- **RL-based project design**
- **Research papers and theses**
- **Industry deployment**
- **Viva and interviews**

This document can be directly used as a **glossary chapter** in a thesis or project report.
