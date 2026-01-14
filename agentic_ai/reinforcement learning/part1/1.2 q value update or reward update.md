# Reinforcement Learning (Beginner)
## Q-Value Updates, Optimism, and Gradient Bandits

---

# 1. How Q(a) Is Updated Incrementally

---

## 1.1 Why Do We Need Incremental Updates?

In real life:
- Data arrives **one step at a time**
- We cannot store all past rewards
- We want to **update knowledge immediately**

So instead of recomputing averages from scratch, we use **incremental updates**.

---

## 1.2 Recall: What Is Q(a)?

\[
Q(a) = \text{expected (average) reward of action } a
\]

It represents:
> “What reward do I usually get if I take action `a`?”

---

## 1.3 Naive (Full Average) Method ❌

If action `a` was taken `n` times:

\[
Q(a) = \frac{R_1 + R_2 + \dots + R_n}{n}
\]

Problem:
- Must store all rewards
- Not scalable

---

## 1.4 Incremental Update Formula ✅

We update Q-value **step by step**:

\[
Q_{new}(a) = Q_{old}(a) + \alpha \big( R - Q_{old}(a) \big)
\]

Where:
- `R` = reward just received
- `α` (alpha) = learning rate (0 < α ≤ 1)

---

## 1.5 Intuition (Very Important)

\[
R - Q_{old}(a) = \text{error}
\]

- If reward is higher than expected → increase Q
- If reward is lower than expected → decrease Q

So we:
> **Move Q slightly toward the new reward**

---

## 1.6 Numerical Example

Assume:
- Initial Q(A) = 4.0
- Learning rate α = 0.1
- New reward R = 5

\[
Q_{new}(A) = 4.0 + 0.1(5 - 4.0) = 4.1
\]

---

## 1.7 Why Incremental Update Is Powerful

✔ Memory efficient  
✔ Fast  
✔ Works online  
✔ Used everywhere in RL  

---

