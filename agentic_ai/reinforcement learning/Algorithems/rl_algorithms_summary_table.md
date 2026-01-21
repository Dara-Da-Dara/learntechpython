# Reinforcement Learning Paradigms – Conceptual Description and Categorized Tables

This document first **conceptually describes the four major Reinforcement Learning paradigms** and then presents the categorized algorithm tables.

---

## A. Classical Reinforcement Learning (Classical RL)

**Classical Reinforcement Learning** refers to early and foundational RL methods that rely on **tabular representations** or simple function approximators. These methods assume relatively **small or discrete state–action spaces** and are grounded in strong mathematical theory.

**Key Characteristics:**
- Uses value functions (V, Q) and/or explicit policies
- Relies on Bellman equations and dynamic programming principles
- Requires full or partial knowledge of states and rewards
- Typically model-free or simple model-based

**Strengths:**
- Strong convergence guarantees
- Easy to analyze and interpret
- Computationally inexpensive for small problems

**Limitations:**
- Poor scalability to large or continuous state spaces
- Requires manual feature engineering

**Typical Domains:**
Robotics control, grid-world problems, operations research, finance, bandit problems

---

## B. Deep Reinforcement Learning (Deep RL)

**Deep Reinforcement Learning** combines classical RL principles with **deep neural networks** to approximate value functions, policies, or environment models. It enables RL to scale to **high-dimensional and unstructured inputs** such as images, audio, and sensor data.

**Key Characteristics:**
- Uses deep neural networks as function approximators
- Handles continuous and high-dimensional state spaces
- Often trained using experience replay and target networks

**Strengths:**
- Scales to complex, real-world problems
- Learns representations automatically from raw data
- Effective in vision-based and continuous control tasks

**Limitations:**
- Sample inefficient
- High computational cost
- Training instability without careful design

**Typical Domains:**
Autonomous driving, robotics, game AI (Atari, AlphaGo), industrial automation

---

## C. Imitation Learning

**Imitation Learning** focuses on learning behavior by **observing expert demonstrations** instead of relying solely on reward signals. The agent attempts to mimic expert actions or infer the underlying reward function.

**Key Characteristics:**
- Learning from expert trajectories
- Can be supervised or inverse reinforcement based
- Reduces exploration cost

**Strengths:**
- Faster learning compared to pure RL
- Suitable when reward design is difficult
- Human-aligned behavior

**Limitations:**
- Requires high-quality expert data
- Sensitive to distribution shift
- Limited generalization beyond demonstrations

**Typical Domains:**
Autonomous driving, robotics manipulation, dialogue systems, healthcare decision support

---

## D. Hierarchical Reinforcement Learning (HRL)

**Hierarchical Reinforcement Learning** decomposes a complex task into **multiple levels of sub-tasks or skills**, allowing agents to plan and learn over different temporal scales.

**Key Characteristics:**
- Uses temporal abstraction (options, skills, subtasks)
- Reduces effective decision horizon
- Often combines planning and learning

**Strengths:**
- Improved scalability
- Better interpretability
- Reusable sub-policies

**Limitations:**
- Manual hierarchy design may be required
- Increased architectural complexity

**Typical Domains:**
Robotics, long-horizon planning, navigation, multi-stage decision-making

---

This document splits the algorithms from both PDFs into **Classical RL, Deep RL, Imitation Learning, and Hierarchical RL** for clearer academic presentation.

---

## 1. Classical Reinforcement Learning

| Algorithm | Year Introduced | Primary Use | Advantages | Limitations | Basic Applications |
|---------|----------------|-------------|------------|-------------|-------------------|
| Multi-Armed Bandit | 1952 | Action selection under uncertainty | Simple, fast learning | No state transitions | Online ads, A/B testing |
| ε-Greedy | 1985 | Exploration–exploitation | Easy to implement | Inefficient exploration | Recommender systems |
| Optimistic Initial Values | 1998 | Encourage exploration | No randomness needed | Sensitive to initialization | Small RL problems |
| Gradient Bandit | 2001 | Preference-based learning | Direct reward optimization | Step-size tuning | Online decision systems |
| Monte Carlo (MC) | 1954 | Episodic value estimation | Unbiased estimates | High variance | Game playing |
| TD(0) | 1988 | Bootstrapped value learning | Low variance | Biased estimates | Robotics control |
| TD(λ) | 1988 | Bias–variance tradeoff | Faster convergence | λ tuning | Prediction tasks |
| Eligibility Traces | 1988 | Credit assignment | Faster learning | Memory overhead | Control systems |
| SARSA | 1996 | On-policy TD control | Safer updates | Slower convergence | Navigation tasks |
| Q-Learning | 1989 | Off-policy optimal control | Guaranteed convergence | Overestimation bias | Routing, games |
| Double Q-Learning | 2010 | Bias reduction | Stable estimates | Extra computation | Control tasks |
| UCB | 2002 | Bandit exploration | Theoretical guarantees | Stationarity assumption | Online ads |
| LSTD | 2003 | Value estimation | Fast convergence | Matrix inversion cost | Finance, control |
| MDP | 1957 | Sequential decision modeling | Strong theory | State explosion | Planning |
| POMDP | 1960 | Partial observability | Models uncertainty | Computationally expensive | Healthcare, robotics |
| MARL | 1999 | Multi-agent learning | Captures interaction | Non-stationarity | Traffic, markets |

---

## 2. Deep Reinforcement Learning

| Algorithm | Year Introduced | Primary Use | Advantages | Limitations | Basic Applications |
|---------|----------------|-------------|------------|-------------|-------------------|
| Deep Q-Network (DQN) | 2013 | High-dimensional control | Handles raw sensory input | Sample inefficient | Atari games |
| Double DQN | 2015 | Reduce value overestimation | More stable learning | Extra computation | Autonomous agents |
| Fitted Q Iteration | 2005 | Batch deep RL | Data efficient | Needs batch data | Industrial control |
| Actor-Critic (Deep) | 1999 | Policy + value learning | Reduced variance | Architecture complexity | Robotics |
| Policy Gradient | 1999 | Continuous action spaces | Direct optimization | Local optima | Control systems |
| PPO | 2017 | Stable policy optimization | Robust & efficient | Approximate constraints | Robotics, games |
| Curiosity-Driven RL | 2017 | Exploration in sparse reward | Encourages novelty | Distracting objectives | Sparse environments |

---

## 3. Imitation Learning

| Algorithm | Year Introduced | Primary Use | Advantages | Limitations | Basic Applications |
|---------|----------------|-------------|------------|-------------|-------------------|
| Behavior Cloning | 1989 | Learn from demonstrations | Simple supervised learning | Error compounding | Autonomous driving |
| Inverse RL | 2000 | Recover reward function | Human-aligned rewards | Ambiguous solutions | Human behavior modeling |
| DAGGER | 2011 | Improve imitation learning | Reduces covariate shift | Requires expert access | Robotics |
| Apprenticeship Learning | 2004 | Safe imitation | Theoretical guarantees | Computational cost | Human–robot interaction |

---

## 4. Hierarchical Reinforcement Learning

| Algorithm | Year Introduced | Primary Use | Advantages | Limitations | Basic Applications |
|---------|----------------|-------------|------------|-------------|-------------------|
| MAXQ Decomposition | 1998 | Task decomposition | Temporal abstraction | Manual hierarchy | Robotics |
| Hierarchical MDP | 1998 | Multi-level planning | Reduced complexity | Design overhead | Planning systems |
| Dyna-Q | 1991 | Learning + planning | Sample efficient | Model bias | Simulated planning |
| Dyna-Q+ | 1995 | Non-stationary exploration | Handles changing env | Extra parameters | Adaptive agents |

---

**Academic Note:** This categorization aligns with standard RL literature (Sutton & Barto; Watkins; Schulman et al.) and is suitable for PhD coursework, exams, and teaching material.

