# REIT Performance Predictor

A web-based implementation inspired by the research paper **"Investment and Finance Committees Composition and Firm Performance: Evidence from U.S. Real Estate Investment Trusts (REITs)"**. This project predicts REIT performance metrics such as **Return on Assets (ROA)** or **Return on Equity (ROE)** using governance-related inputs.

---

## Research Motivation and Overview

The research paper investigates the effect of governance structures, specifically the percentage of **inside directors** on finance and investment committees, on the performance of U.S. Real Estate Investment Trusts (REITs). Using regression models, it demonstrates that:
- A higher percentage of inside directors correlates positively with performance metrics like ROA for large REITs listed in the S&P 500.
- Conversely, this relationship can be negative for smaller, non-S&P 500 REITs.

This project translates these findings into a **machine learning-based REIT performance predictor**. Users can input governance-related features such as `% Inside Directors`, **REIT Size**, **Lifecycle Stage**, and **S&P 500 Status** to predict performance.

---

## Advantages, Disadvantages, and Nuances

### Advantages
1. **Actionable Insights**: Helps stakeholders assess how governance structures impact REIT performance.
2. **Interactive**: Provides real-time performance predictions based on user inputs.
3. **Scalable**: Can be enhanced with real-world data sources like Zillow API or financial databases.

### Disadvantages
1. **Synthetic Data**: The current model uses synthetic data due to limited access to real-world REIT datasets.
2. **Simplified Scope**: Focuses only on a subset of governance factors without including external market variables.

### Nuances
- The relationship between inside directors and performance is **context-dependent**. While beneficial for S&P 500 REITs, it may be detrimental to smaller REITs, highlighting the importance of tailoring board structures.

---

## Key Takeaways and Novelty

- **Key Takeaway**: Governance structures, specifically the percentage of inside directors on committees, significantly impact REIT performance.
- **Novelty**: This implementation bridges theoretical findings from the paper with practical, interactive predictions using machine learning.

---

## Mathematical Model

The predictive model is a **feed-forward neural network** trained on governance-related features to predict ROA/ROE.

### Features
Let \( X = [x_1, x_2, x_3, x_4] \), where:
- \( x_1 \): % Inside Directors.
- \( x_2 \): REIT Size (e.g., small, medium, large).
- \( x_3 \): Lifecycle Stage (1 = early, 2 = growth, 3 = mature).
- \( x_4 \): S&P 500 Status (0 = No, 1 = Yes).

### Model Architecture
- **Input Layer**: 4 features.
- **Hidden Layers**:
  - Layer 1: 64 neurons (ReLU activation).
  - Layer 2: 32 neurons (ReLU activation).
- **Output Layer**: Single neuron for the predicted ROA/ROE.

The output \( y \) is calculated as:
\[
y = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot X + b_1) + b_2) + b_3
\]
Where \( W_i \) and \( b_i \) are the learned weights and biases.

### Loss Function
The model minimizes **Mean Squared Error (MSE)**:
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]
Where \( y_i \) is the true performance and \( \hat{y}_i \) is the predicted performance.

---

## Model Training

### Training Process
- **Dataset**: A synthetic dataset was generated to simulate the relationship between governance factors and performance.
- **Epochs**: The model was trained for **3 epochs**, which was sufficient for the loss function to converge. Training for 3 epochs balances computational efficiency with prediction accuracy.

### Results
- **Final Loss**: The MSE after 3 epochs was approximately **0.0016**, indicating good convergence.
- **Predictions**: The model accurately predicted synthetic ROA/ROE values, as validated against test data.

---

## Project Workflow

1. **Input Governance Features**:
   Users provide governance-related inputs, such as `% Inside Directors`, **REIT Size**, and **Lifecycle Stage**.

2. **Backend Prediction**:
   The Flask backend loads the trained REIT model, processes user inputs, and returns the predicted performance.

3. **Interactive Frontend**:
   The web-based interface allows users to enter inputs and view predictions in real time.




