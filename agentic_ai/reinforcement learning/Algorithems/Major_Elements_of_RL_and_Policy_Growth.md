
# Major Elements of Reinforcement Learning (RL)
**Focus: Agent, Policy, and Growth of Agent Policy Across Algorithms**

---

## 1. Major Elements of Reinforcement Learning

Reinforcement Learning is a paradigm of machine learning where an **agent** learns to make decisions by interacting with an **environment** to maximize cumulative reward.

### 1.1 Agent
- The **agent** is the learner or decision-maker.
- It observes the environment, selects actions, and learns from feedback.
- Examples: robot, game-playing AI, recommendation system.

### 1.2 Environment
- Everything external to the agent.
- Provides states and rewards in response to agent actions.

### 1.3 State (S)
- A representation of the current situation of the environment.
- Can be fully observable (MDP) or partially observable (POMDP).

### 1.4 Action (A)
- The set of all possible moves the agent can take.
- Can be discrete or continuous.

### 1.5 Reward (R)
- Scalar feedback signal from the environment.
- Guides the learning process.
- Objective: maximize expected cumulative (discounted) reward.

### 1.6 Policy (π)
- A policy defines the agent’s behavior.
- Mapping from states to actions.
- Central component of RL learning.

### 1.7 Value Function
- Estimates how good a state or action is.
- State-value: V(s)
- Action-value: Q(s, a)

### 1.8 Model (Optional)
- Predicts next state and reward.
- Used in model-based RL.

---

## 2. Agent Policy in Reinforcement Learning

### 2.1 Definition of Policy
A **policy (π)** specifies the probability of taking action *a* in state *s*:

π(a | s)

### 2.2 Types of Policies

**Deterministic Policy**
- Always selects the same action for a given state.
- Example: π(s) = a

**Stochastic Policy**
- Selects actions based on probabilities.
- Example: π(a|s) = 0.7 left, 0.3 right

### 2.3 Role of Policy
- Controls agent behavior
- Balances exploration and exploitation
- Improves through learning

---

## 3. Growth of Agent Policy Across Different RL Algorithms

Policy growth refers to how the agent’s decision-making improves over time using learning algorithms.

---

## 4. Policy Growth in Value-Based Algorithms

### 4.1 Q-Learning (Off-policy)

**Mechanism**
- Learns Q(s, a)
- Policy is derived indirectly using greedy selection

**Policy Growth**
1. Initialize Q-table arbitrarily
2. Explore using ε-greedy policy
3. Update Q-values via TD learning
4. Policy improves as Q-values converge

**Key Property**
- Learns optimal policy independent of behavior policy

---

### 4.2 SARSA (On-policy)

**Mechanism**
- Learns Q(s, a) using current policy

**Policy Growth**
- Policy evolves gradually with learning
- Safer and more conservative than Q-learning

**Key Property**
- Policy improvement tied to actual actions taken

---

## 5. Policy Growth in Policy-Based Algorithms

### 5.1 REINFORCE (Monte Carlo Policy Gradient)

**Mechanism**
- Directly parameterizes policy πθ
- Updates policy using episode returns

**Policy Growth**
1. Start with random policy
2. Sample trajectories
3. Increase probability of rewarding actions
4. Gradual improvement with high variance

**Key Property**
- No value function required

---

## 6. Policy Growth in Actor-Critic Algorithms

### 6.1 Actor-Critic

**Components**
- Actor: updates policy
- Critic: evaluates policy using value function

**Policy Growth**
- Actor improves policy using critic feedback
- Faster and more stable than pure policy gradients

---

### 6.2 Proximal Policy Optimization (PPO)

**Mechanism**
- Constrains policy updates
- Uses clipped objective

**Policy Growth**
- Smooth and stable improvement
- Prevents destructive large updates

**Key Property**
- Widely used in deep RL

---

## 7. Policy Growth in Model-Based RL

### 7.1 Dyna-Q

**Mechanism**
- Learns model + Q-values

**Policy Growth**
- Faster learning using simulated experiences
- Efficient policy refinement

---

## 8. Policy Growth in Multi-Agent RL (MARL)

### 8.1 Independent Q-Learning
- Each agent learns its own policy

### 8.2 Centralized Training, Decentralized Execution
- Policies grow using shared critic information

**Challenge**
- Non-stationary environment

---

## 9. Summary Table

| Algorithm Type | Policy Representation | Policy Growth Style |
|---------------|----------------------|--------------------|
| Q-Learning | Implicit (Q-table) | Greedy improvement |
| SARSA | Implicit | Gradual, safe |
| REINFORCE | Explicit | Probabilistic |
| Actor-Critic | Explicit | Stable, guided |
| PPO | Explicit | Constrained |
| Dyna-Q | Implicit | Accelerated |
| MARL | Multiple policies | Interdependent |

---

## 10. Key Takeaways
- Policy is the core of RL intelligence
- Value-based methods derive policy indirectly
- Policy-based methods learn policy directly
- Actor-Critic balances stability and efficiency
- Policy growth reflects learning, adaptation, and optimization

---

**End of Document**
