# On-Policy vs Off-Policy Learning and Model-Based vs Model-Free Reinforcement Learning


- On-Policy Learning
- Off-Policy Learning
- Model-Based Reinforcement Learning
- Model-Free Reinforcement Learning

.

---

## 1. On-Policy Learning

### Practical Examples and Real-Time Applications

**Example 1: Robot Navigation in Crowded Environments**  
A service robot in a hospital learns navigation using **SARSA**. Since unsafe actions (collisions) are costly, the robot learns only from the actions it actually takes, ensuring safer learning.

**Real-Time Applications**
- Service robots in hospitals
- Autonomous wheelchairs
- Online recommendation systems with safety constraints

---

## 1. On-Policy Learning

### Meaning
**On-policy learning** refers to learning a policy while **following the same policy** to interact with the environment.

> *The agent learns from actions it actually performs.*

### Formal Definition
If the agent follows policy \( \pi \), it updates the value of the **same policy**:

\[
\text{Behavior Policy} = \text{Target Policy} = \pi
\]

### Key Characteristics
- Learns from current behavior
- Naturally incorporates exploration
- More stable but slower learning

### Common On-Policy Algorithms
- SARSA
- On-policy Monte Carlo
- Policy Gradient Methods
- Proximal Policy Optimization (PPO)

### Real-Life Example
A **student learning to drive** improves only based on their own driving experience and mistakes.

### Advantages
- Stable learning
- Safer in uncertain environments

### Limitations
- Cannot reuse old data efficiently
- Data inefficient

---

## 2. Off-Policy Learning

### Practical Examples and Real-Time Applications

**Example 1: Autonomous Driving (Simulation + Real Data)**  
Self-driving cars learn optimal driving policies using **Q-learning/DQN** from recorded driving data and simulations, without executing every risky maneuver.

**Real-Time Applications**
- Autonomous vehicles (Waymo, Tesla simulations)
- Robotics learning from logs
- Ad placement systems using historical data

---

## 2. Off-Policy Learning

### Meaning
**Off-policy learning** refers to learning a policy that is **different from the policy used to generate data**.

> *The agent learns from experiences it did not directly generate.*

### Formal Definition
\[
\text{Behavior Policy } \mu \neq \text{ Target Policy } \pi
\]

### Key Characteristics
- Can reuse past or external data
- Faster learning
- Less stable without proper techniques

### Common Off-Policy Algorithms
- Q-Learning
- Deep Q-Networks (DQN)
- Double DQN
- Deep Deterministic Policy Gradient (DDPG)

### Real-Life Example
A **learner driver watching recorded driving videos** learns without making all mistakes personally.

### Advantages
- High sample efficiency
- Enables learning from demonstrations

### Limitations
- Potential instability
- Requires corrections (target networks, importance sampling)

---

## 3. Model-Based Reinforcement Learning

### Practical Examples and Real-Time Applications

**Example 1: Industrial Process Control**  
An agent learns a model of a chemical plant and simulates outcomes before adjusting control parameters, reducing costly real-world experimentation.

**Real-Time Applications**
- Robotics motion planning
- Energy grid optimization
- Financial portfolio planning

---

## 3. Model-Based Reinforcement Learning

### Meaning
**Model-based RL** involves learning or using an explicit **model of the environment** to plan actions.

> *The agent thinks ahead before acting.*

### Environment Model Components
- Transition Model: \( P(s' | s, a) \)
- Reward Model: \( R(s, a) \)

### Key Characteristics
- Combines learning with planning
- Sample efficient
- Computationally intensive

### Common Model-Based Methods
- Dynamic Programming
- Dyna-Q
- Model Predictive Control
- World Models

### Real-Life Example
A **chess player** evaluates future moves before making a decision.

### Advantages
- Requires fewer real interactions
- Interpretable decision-making

### Limitations
- Inaccurate models can degrade performance
- High computational cost

---

## 4. Model-Free Reinforcement Learning

### Practical Examples and Real-Time Applications

**Example 1: Game Playing Agents**  
AlphaGo and Atari-playing agents learn directly from gameplay experience without modeling game dynamics.

**Real-Time Applications**
- Video game AI
- Recommendation engines
- Traffic signal control

---

## 4. Model-Free Reinforceme

### Meaning
**Model-free RL** learns optimal behavior **directly from interaction**, without any knowledge of environment dynamics.

> *Learning by trial and error.*

### Key Characteristics
- No explicit model of the environment
- Simple and flexible
- Data intensive

### Common Model-Free Algorithms
- Q-Learning
- SARSA
- Policy Gradient
- PPO
- DQN

### Real-Life Example
A **child learning to ride a bicycle** improves balance through repeated attempts.

### Advantages
- Easy to implement
- Works in unknown environments

### Limitations
- Requires large amounts of data
- Less interpretable

---

## 5. Summary Comparison Tables

### On-Policy vs Off-Policy Learning

| Aspect | On-Policy | Off-Policy |
|------|----------|-----------|
| Policy learned | Same as behavior | Different from behavior |
| Data reuse | Not possible | Possible |
| Stability | High | Medium–Low |
| Learning speed | Slower | Faster |
| Example | SARSA | Q-Learning |

---

### Model-Based vs Model-Free Learning

| Aspect | Model-Based RL | Model-Free RL |
|------|---------------|---------------|
| Environment model | Required | Not required |
| Planning | Yes | No |
| Sample efficiency | High | Low |
| Complexity | High | Low |
| Example | Dyna-Q | DQN |

---

## 6. One-Line Exam Answers

- **On-policy learning:** The agent learns the same policy it follows.
- **Off-policy learning:** The agent learns a different policy from the one used to act.
- **Model-based RL:** The agent uses an environment model to plan actions.
- **Model-free RL:** The agent learns directly from interaction without a model.

---

This Markdown file is **ready for download, submission, or conversion to PDF/DOCX**.

