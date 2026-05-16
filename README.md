# California-Housing-Price-Prediction-MLP
# House Price Prediction using Multilayer Perceptron (MLP)

# 1. Problem Description
This project aims to predict house prices using the standard **California Housing Dataset**. Since the target variable (house value) is continuous, this is a **Regression** task solved using Deep Learning (Neural Networks - Multilayer Perceptron).

# 2. Dataset & Preprocessing (Requirement 2)
* **Dataset Name:** California Housing Dataset (fetched directly via Scikit-Learn).
* **Preprocessing steps applied:**
  * Feature and target partitioning ($X$ and $y$).
  * Train-Test Splitting (80% training, 20% testing).
  * Feature Scaling using `StandardScaler` to normalize data for the MLP input layer.

# 3. Model Architecture (Requirement 3)
Implemented a Multilayer Perceptron (MLP) using TensorFlow/Keras. The network contains an Input layer, multiple Hidden layers, and a Single-neuron Output layer with a `linear` activation function suitable for regression tasks.

# 4. Experiments & Results Comparison (Requirement 6 & 7)
Two mandatory experiments were conducted by changing the activation functions, number of neurons, and the learning rate.

| Experiment | Architecture (Hidden Layers) | Activation | Learning Rate | Regularization | Test Loss (MSE) | Final MAE |
|------------|------------------------------|------------|---------------|----------------|-----------------|-----------|
| **Exp 1: Base Model** | [32, 16] | `relu` | 0.01 | None | **0.2969** | 0.3708 |
| **Exp 2: Enhanced** | [64, 32, 16] | `tanh` | 0.001 | Dropout (0.2) | **0.3099** | 0.3769 |

# Visualizations (Training vs Validation Loss)
![Loss Curves](loss_curves.png)

# Discussion for Discussion/Interview:
* **Experiment 1** achieved a slightly better Test MSE (**0.2969**), showing that a simpler architecture with `relu` activation converges very efficiently on this dataset.
* **Experiment 2** utilized `tanh` activation and a smaller learning rate, along with **Dropout (0.2)** as an optional enhancement to stabilize the validation loss curve, resulting in a highly smooth and stable learning curve.

# 5. How to Run the Project
1. Clone this repository.
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn tensorflow`
3. Run the python script: `python neural_network_project.py`
