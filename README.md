# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Problem Statement
Build and analyze a feed-forward neural network on the **Customer Churn** dataset to predict whether a customer will churn. The goal is to demonstrate how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

---

## Dataset
- **File:** `customer_churn_nn.csv`
- **Rows:** 2000 | **Columns:** 17
- **Target:** `churn` (0 = Retained, 1 = Churned)
- **Class Distribution:** 1969 retained (98.45%), 31 churned (1.55%) — heavily imbalanced
- **Categorical features:** `region`, `plan_type`, `contract_type`, `payment_method`
- **Numerical features:** `tenure_months`, `monthly_charges_inr`, `avg_login_days_per_month`, `support_tickets_last_90_days`, `payment_delay_days`, `data_usage_gb`, `satisfaction_score`, `last_complaint_days_ago`, `discount_percent`, `autopay_enabled`, `referral_count`

---

## Tasks Covered

| Task | Description |
|------|-------------|
| Task 1 | Dataset understanding: shape, types, missing values, statistical summary, target distribution |
| Task 2 | Preprocessing: label encoding, standard scaling, train/test split (80/20) |
| Task 3 | Model building: feed-forward MLP with two hidden layers (64 → 32), ReLU, Adam optimizer |
| Task 4 | Training and evaluation: accuracy, loss, confusion matrix, classification report |
| Task 5 | Hyperparameter experiments: 5 configurations compared by train/test accuracy and loss |
| Task 6 | Final reflection: weights & biases, activation functions, learning rate effects, overfitting |

---

## Model Architecture (Baseline)

```
Input Layer        →  15 features
Hidden Layer 1     →  64 neurons, ReLU
Hidden Layer 2     →  32 neurons, ReLU
Output Layer       →  1 neuron (binary: churn/no-churn)
Loss Function      →  Binary Cross-Entropy
Optimizer          →  Adam (lr=0.001)
```

---

## Hyperparameter Experiments

| Experiment | Hidden Layers | Activation | Learning Rate | Batch Size | Test Accuracy |
|---|---|---|---|---|---|
| Baseline | (64, 32) | ReLU | 0.001 | 32 | 97.50% |
| Deeper Network | (128, 64, 32) | ReLU | 0.001 | 32 | 98.00% |
| Higher LR | (64, 32) | ReLU | 0.010 | 32 | 97.25% |
| Tanh Activation | (64, 32) | Tanh | 0.001 | 32 | 97.00% |
| Large Batch | (64, 32) | ReLU | 0.001 | 128 | 97.75% |

> **Note:** High accuracy is largely driven by class imbalance. The model tends to predict the majority class (Retained). Real-world improvements would require SMOTE or class weighting.

---

## Key Findings

- **Weights & Biases:** Weights scale input signal strength; biases shift activations. Both are updated via backpropagation.
- **Activation Functions:** ReLU and Tanh introduce non-linearity, enabling the network to model complex patterns.
- **Learning Rate:** A high LR (0.01) caused slightly worse test accuracy; too low a LR would cause slow convergence.
- **Overfitting/Underfitting:** Training accuracy hit 100% while test accuracy stayed ~97–98%, indicating mild overfitting on the minority class (underfitting for churn detection).

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md                     ← This file
├── analysis.py                   ← Full Python script (all 6 tasks)
├── requirements.txt              ← Python dependencies
└── results/
    ├── target_distribution.png   ← Task 1: Churn class distribution chart
    ├── evaluation_outputs.png    ← Task 4: Confusion matrix
    ├── loss_curve.png            ← Task 4: Training loss curve
    ├── model_comparison_table.csv← Task 5: Experiment results table
    └── model_comparison_table.png← Task 5: Bar chart comparison
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tasks
python analysis.py
```

---

## Requirements
See `requirements.txt` for the full list of dependencies.
