# Day 1: Introduction to Machine Learning

## 1. What is Machine Learning?
Machine Learning (ML) is a subset of Artificial Intelligence (AI) that allows systems to learn from data and make decisions or predictions without being explicitly programmed.

**Key Idea:** Instead of writing rules manually, we teach a computer by providing data.

---

## 2. Types of Machine Learning

### 2.1 Supervised Learning
- The model is trained on **labeled data** (input + correct output).
- Goal: Predict output for new, unseen inputs.

**Examples:**
- Predicting house prices based on features like size, location.
- Email spam detection.

**Common Algorithms:**
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- Neural Networks

---

### 2.2 Unsupervised Learning
- The model is trained on **unlabeled data** (only inputs, no outputs).
- Goal: Find hidden patterns or structure in data.

**Examples:**
- Customer segmentation
- Market basket analysis

**Common Algorithms:**
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)

---

### 2.3 Reinforcement Learning
- The model **learns by interacting with an environment**.
- It receives **rewards** for good actions and **penalties** for bad actions.

**Examples:**
- Self-driving cars
- Game AI (chess, Go)

---

## 3. Basic ML Workflow

1. **Collect Data:** Gather the dataset needed for the problem.
2. **Preprocess Data:** Clean data, handle missing values, encode categorical variables.
3. **Split Data:** Divide into training and testing sets (commonly 80/20).
4. **Train Model:** Use training data to teach the model.
5. **Evaluate Model:** Test model on unseen data to check performance.
6. **Improve Model:** Tune parameters, try different algorithms.
7. **Deploy Model:** Use the model in real-world applications.

---

## 4. Applications of Machine Learning

- **Healthcare:** Disease prediction, medical imaging analysis
- **Finance:** Fraud detection, stock price prediction
- **Retail:** Customer segmentation, recommendation systems
- **Technology:** Voice assistants, self-driving cars
- **Agriculture:** Crop yield prediction, pest detection

---

## 5. Key Terms in Machine Learning

- **Feature:** Input variable used to make predictions
- **Label/Target:** Output variable the model predicts
- **Model:** Mathematical representation of the relationship between inputs and outputs
- **Training:** Process of teaching the model using data
- **Prediction:** Output produced by the model
- **Overfitting:** Model performs well on training data but poorly on new data
- **Underfitting:** Model is too simple to capture patterns in data

---

## 6. Python Libraries for ML

- `numpy` – Numerical operations
- `pandas` – Data manipulation
- `scikit-learn` – Basic ML algorithms
- `matplotlib` / `seaborn` – Data visualization
- `tensorflow` / `pytorch` – Deep learning

---

## Summary
Machine Learning enables computers to **learn from data** and make intelligent decisions. It is divided mainly into **Supervised, Unsupervised, and Reinforcement Learning**, with applications across industries. The ML workflow involves **data collection, preprocessing, training, evaluation, and deployment**.

# Day 2: Supervised Machine Learning

## 1. What is Supervised Learning?
Supervised Learning is a type of Machine Learning where the model is trained using **labeled data**.  
- Labeled data means each input comes with the correct output (target/label).  
- The model **learns the relationship** between input features and the target.

**Goal:** Predict the output for new, unseen inputs.

---

## 2. Types of Supervised Learning

### 2.1 Regression
- **Used for:** Predicting continuous numeric values.
- **Output:** A real number.
- **Examples:**
  - Predicting house prices
  - Predicting temperature
  - Predicting sales revenue

**Common Algorithms:**
- Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)

---

### 2.2 Classification
- **Used for:** Predicting discrete categories or classes.
- **Output:** A label or class.
- **Examples:**
  - Email spam detection (spam/not spam)
  - Disease diagnosis (yes/no)
  - Handwriting recognition (digits 0–9)

**Common Algorithms:**
- Logistic Regression
- Decision Trees
- Random Forest
- k-Nearest Neighbors (k-NN)
- Support Vector Machines (SVM)

---

## 3. Steps in Supervised Learning

1. **Collect Data:** Gather a dataset with input features and corresponding labels.
2. **Preprocess Data:** Clean data, handle missing values, normalize/scale features, encode categorical variables.
3. **Split Data:** Divide dataset into:
   - Training set (e.g., 70–80%)
   - Testing set (e.g., 20–30%)
4. **Train Model:** Fit the model on the training set.
5. **Evaluate Model:** Check performance on the test set using metrics.
6. **Tune Model:** Optimize hyperparameters or try different algorithms.
7. **Predict:** Use the model for new unseen data.

---

## 4. Evaluation Metrics

### 4.1 For Regression
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

### 4.2 For Classification
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---

## 5. Python Example: Simple Linear Regression

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {'Size': [1000, 1500, 2000, 2500, 3000],
        'Price': [200000, 250000, 300000, 350000, 400000]}
df = pd.DataFrame(data)

# Features and target
X = df[['Size']]  # Feature
y = df['Price']   # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


---
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset (for example, Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # weighted for multi-class
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
---

