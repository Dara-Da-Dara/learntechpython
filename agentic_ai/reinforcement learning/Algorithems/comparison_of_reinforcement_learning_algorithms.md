# Comparison of Reinforcement Learning Algorithms

This document provides a **clear comparative study of major Reinforcement Learning algorithms** across **Classical RL, Deep RL, Imitation Learning, and Hierarchical RL**. The comparison focuses on **learning style, data requirements, stability, scalability, and real-world suitability**, making it ideal for **exam answers, viva, and coursework**.

---

## 1. Classical Reinforcement Learning Algorithms

| Algorithm | Learning Type | Policy Type | Model Usage | Key Strengths | Key Limitations | Typical Applications |
|----------|--------------|-------------|-------------|---------------|-----------------|---------------------|
| Q-Learning | Value-based | Off-policy | Model-free | Converges to optimal policy | Overestimation bias | Routing, games |
| SARSA | Value-based | On-policy | Model-free | Safer learning | Slower convergence | Navigation, robotics |
| Monte Carlo | Value-based | On/Off-policy | Model-free | Unbiased estimates | High variance | Game playing |
| TD(0) | Value-based | On-policy | Model-free | Low variance | Biased estimates | Prediction tasks |
| UCB (Bandits) | Value-based | Off-policy | Model-free | Theoretical guarantees | Assumes stationarity | Online ads |

---

## 2. Deep Reinforcement Learning Algorithms

| Algorithm | Network Type | Action Space | Stability | Sample Efficiency | Key Advantage | Key Limitation | Applications |
|---------|--------------|--------------|-----------|------------------|---------------|----------------|-------------|
| DQN | Value Network | Discrete | Medium | Low | Handles high-dimensional states | Needs large data | Atari games |
| Double DQN | Value Network | Discrete | High | Low | Reduces overestimation | Extra computation | Autonomous agents |
| Actor-Critic | Policy + Value | Continuous | Medium | Medium | Reduced variance | Architecture complexity | Robotics |
| PPO | Policy Gradient | Continuous/Discrete | High | Medium | Stable updates | Approximate guarantees | Robotics, games |
| Curiosity-Driven RL | Hybrid | Any | Medium | Low | Solves sparse rewards | Distracting objectives | Exploration tasks |

---

## 3. Imitation Learning Algorithms

| Algorithm | Learning Paradigm | Data Requirement | Reward Needed | Strengths | Limitations | Applications |
|----------|------------------|------------------|---------------|-----------|-------------|-------------|
| Behavior Cloning | Supervised | Expert trajectories | No | Simple, fast | Compounding error | Autonomous driving |
| DAgger | Interactive IL | Expert + agent data | No | Reduces distribution shift | Requires expert | Robotics |
| Inverse RL | Reward inference | Expert trajectories | Implicit | Captures intent | Computationally heavy | Preference learning |
| Apprenticeship Learning | IRL-based | Expert trajectories | Implicit | Theoretical guarantees | Slow training | Human–robot interaction |

---

## 4. Hierarchical Reinforcement Learning Algorithms

| Algorithm | Hierarchy Type | Temporal Abstraction | Scalability | Strengths | Limitations | Applications |
|----------|---------------|---------------------|-------------|-----------|-------------|-------------|
| MAXQ | Task decomposition | High | Medium | Interpretable hierarchy | Manual design | Robotics |
| Options Framework | Skill-based | High | High | Reusable skills | Option discovery | Navigation |
| Hierarchical DQN | Deep + HRL | High | High | Handles long horizons | Training complexity | Complex games |
| Feudal RL | Manager–Worker | Very High | High | Clear abstraction | Credit assignment | Large-scale control |

---

## 5. High-Level Comparison Across Paradigms

| Aspect | Classical RL | Deep RL | Imitation Learning | Hierarchical RL |
|------|--------------|---------|-------------------|-----------------|
| State Space Size | Small | Large | Medium–Large | Very Large |
| Data Requirement | Low | Very High | Medium | High |
| Reward Dependence | High | High | Low / None | Medium |
| Interpretability | High | Low | Medium | High |
| Training Cost | Low | Very High | Medium | High |
| Safety During Training | Low | Low | High | Medium |

---

## Exam-Oriented Insight

- **Q-learning vs SARSA** → Off-policy vs On-policy safety
- **DQN vs PPO** → Value-based vs Policy-based stability
- **Behavior Cloning vs DAgger** → Static vs interactive imitation
- **Flat RL vs HRL** → Short-horizon vs long-horizon decision making

This comparison is **ready for 10–15 mark answers, viva discussions, and teaching notes**.