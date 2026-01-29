# Error Metrics in Classification

This document explains **confusion matrix**, **precision**, **recall**, and **F1-score** with clear definitions and worked examples.

---

## 1. Confusion Matrix

A **confusion matrix** is a table used to evaluate the performance of a classification model by comparing **actual labels** with **predicted labels**.

### Binary Classification Confusion Matrix

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Meaning of Terms
- **True Positive (TP)**: Model predicts positive, and it is actually positive
- **True Negative (TN)**: Model predicts negative, and it is actually negative
- **False Positive (FP)**: Model predicts positive, but it is actually negative (Type I error)
- **False Negative (FN)**: Model predicts negative, but it is actually positive (Type II error)

---

## 2. Precision

### Definition
**Precision** measures how many of the predicted positive instances are actually positive.

### Formula
```
Precision = TP / (TP + FP)
```

### Interpretation
- High precision means **few false positives**
- Important when the cost of false positives is high

### Example
Assume:
- TP = 40
- FP = 10

```
Precision = 40 / (40 + 10) = 0.80
```

So, **80%** of predicted positives are correct.

---

## 3. Recall (Sensitivity or True Positive Rate)

### Definition
**Recall** measures how many actual positive instances were correctly identified.

### Formula
```
Recall = TP / (TP + FN)
```

### Interpretation
- High recall means **few false negatives**
- Important when missing a positive case is costly

### Example
Assume:
- TP = 40
- FN = 20

```
Recall = 40 / (40 + 20) = 0.67
```

So, the model identifies **67%** of actual positives.

---

## 4. F1 Score

### Definition
The **F1-score** is the harmonic mean of precision and recall. It balances both metrics.

### Formula
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Interpretation
- Best used when you need a balance between precision and recall
- Especially useful for **imbalanced datasets**

### Example
Using:
- Precision = 0.80
- Recall = 0.67

```
F1 = 2 × (0.80 × 0.67) / (0.80 + 0.67)
   = 0.73
```

The F1-score is **0.73**.

---

## 5. Combined Example

Assume the confusion matrix values:
- TP = 50
- FP = 5
- FN = 10
- TN = 35

### Calculations

```
Precision = 50 / (50 + 5) = 0.91
Recall    = 50 / (50 + 10) = 0.83
F1 Score  = 2 × (0.91 × 0.83) / (0.91 + 0.83) = 0.87
```

---

## 6. When to Use Which Metric

| Metric    | Best Used When |
|-----------|----------------|
| Precision | False positives are costly (e.g., spam detection) |
| Recall    | False negatives are costly (e.g., disease detection) |
| F1 Score  | Need balance between precision & recall |

---

## 7. Practical Meaning (Real-World Example: Spam Email Filter)

- **True Positive (TP):** Spam email correctly moved to the spam folder
- **False Positive (FP):** Important (legitimate) email wrongly sent to spam
- **True Negative (TN):** Important email correctly kept in the inbox
- **False Negative (FN):** Spam email wrongly delivered to the inbox

---

## 8. Key Takeaways
- Confusion matrix is the foundation for all classification metrics
- Precision focuses on prediction quality
- Recall focuses on coverage of actual positives
- F1-score balances precision and recall

---

## 9. Practical Code Example (Python)

Below is a simple Python example showing how **TP, FP, TN, FN**, **Precision**, **Recall**, and **F1-score** are calculated for a spam email classifier.

```python
# Actual labels (1 = Spam, 0 = Not Spam)
y_true = [1, 0, 1, 1, 0, 0, 1, 0]

# Predicted labels
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

# Initialize counts
TP = FP = TN = FN = 0

for actual, predicted in zip(y_true, y_pred):
    if actual == 1 and predicted == 1:
        TP += 1
    elif actual == 0 and predicted == 1:
        FP += 1
    elif actual == 0 and predicted == 0:
        TN += 1
    elif actual == 1 and predicted == 0:
        FN += 1

# Metrics calculation
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("True Positive:", TP)
print("False Positive:", FP)
print("True Negative:", TN)
print("False Negative:", FN)
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1_score, 2))
```

### Output Explanation
- TP: Spam emails correctly detected
- FP: Legitimate emails wrongly marked as spam
- TN: Legitimate emails correctly identified
- FN: Spam emails missed by the model

---

**End of Document**

