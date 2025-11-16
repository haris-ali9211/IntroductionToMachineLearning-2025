"""
===============================================================
LOGISTIC REGRESSION ON DATASET 2 (NUMPY IMPLEMENTATION)

NetSpecs / Overview
-------------------
This script implements a binary classifier using Logistic Regression
from scratch (NumPy only) and applies it to "dataset2.npz".

It covers three main tasks:

Task (a): Train a Logistic Regression model and report the test accuracy.
Task (b): Visualize the linear decision boundary learned by the model.
Task (c): Study how the Negative Log-Likelihood (NLL) changes over
          iterations for different learning rates.

Files required:
- dataset2.npz  (must contain: X_train, X_test, y_train, y_test)

Main Concepts:
- Logistic Regression (linear classifier + sigmoid)
- Gradient Descent optimization
- Negative Log-Likelihood (NLL) as loss function
- Effect of learning rate on convergence behavior
===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ===============================================================
# Logistic Regression (Numpy Only, Binary Classification)
# ===============================================================

class LogisticRegression:
    """
    Simple Logistic Regression implementation using NumPy only.

    - Uses gradient descent to learn weights w and bias b.
    - Optimizes Negative Log-Likelihood (NLL).
    - Supports probability prediction and hard class prediction.
    """

    def __init__(self):
        # w: weight vector of shape (n_features,)
        # b: scalar bias term
        # nll_history: stores NLL at each iteration for analysis/plotting
        self.w = None
        self.b = 0.0
        self.nll_history = []

    def _sigmoid(self, z):
        """
        Sigmoid function:
        Maps any real value z to (0, 1), interpreted as probability.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def _nll(self, y_true, y_prob):
        """
        Negative Log-Likelihood for binary classification.

        y_true: ground truth labels in {0, 1}
        y_prob: predicted probabilities for class 1
        """
        eps = 1e-15  # small constant to avoid log(0)
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        )

    def fit(self, X, y, lr=0.0001, max_iter=1000, return_nll=False):
        """
        Train the logistic regression model using gradient descent.

        X         : training data, shape (n_samples, n_features)
        y         : labels in {0, 1}, shape (n_samples,)
        lr        : learning rate (step size for gradient descent)
        max_iter  : number of gradient descent iterations
        return_nll: if True, returns the NLL history for plotting

        The function updates self.w and self.b in-place.
        """
        n_samples, n_features = X.shape
        y = y.astype(float)  # ensure numeric type

        # Initialize parameters to zero
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.nll_history = []

        # Gradient descent loop
        for i in range(max_iter):
            # Linear combination + bias
            z = X.dot(self.w) + self.b

            # Predicted probabilities via sigmoid
            y_hat = self._sigmoid(z)

            # Compute current loss (Negative Log-Likelihood)
            nll = self._nll(y, y_hat)
            self.nll_history.append(nll)

            # Compute gradients of loss w.r.t w and b
            error = (y_hat - y)                 # shape: (n_samples,)
            grad_w = X.T.dot(error) / n_samples # shape: (n_features,)
            grad_b = np.mean(error)             # scalar

            # Gradient descent parameter update
            self.w -= lr * grad_w
            self.b -= lr * grad_b

        if return_nll:
            return self.nll_history

    def predict_proba(self, X):
        """
        Returns predicted probabilities for class 1.
        """
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        """
        Returns hard class predictions in {0, 1}
        using 0.5 as decision threshold.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)


# ===============================================================
# Load Dataset 2
# ===============================================================

# dataset2.npz is expected to contain:
# - X_train: training features
# - X_test : test features
# - y_train: training labels
# - y_test : test labels
dataset = np.load("dataset2.npz")
X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_train = dataset["y_train"]
y_test = dataset["y_test"]


# ===============================================================
# Task (a): Train Logistic Regression and Report Test Accuracy
# ===============================================================

# Create model instance
logreg = LogisticRegression()

# Train with chosen learning rate and iterations
logreg.fit(X_train, y_train, lr=1e-4, max_iter=1000)

# Predict on test set and compute accuracy
y_pred = logreg.predict(X_test)
print("Task (a) - Test Accuracy:", accuracy_score(y_test, y_pred))


# ===============================================================
# Task (b): Decision Boundary Plot
# ===============================================================

def plot_decision_boundary(model, X, y):
    """
    Plot the linear decision boundary learned by logistic regression
    along with the training data points.

    Assumes X has exactly 2 features so we can plot in 2D.
    """

    # Extract learned weights and bias
    w = model.w
    b = model.b

    # Create a range of x-values for plotting the decision line
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 200)

    # Decision boundary equation:
    #   w1 * x + w2 * y + b = 0
    #   => y = -(w1 * x + b) / w2
    if abs(w[1]) < 1e-8:  # avoid division by zero for nearly vertical line
        y_vals = np.zeros_like(x_vals)
    else:
        y_vals = -(w[0] * x_vals + b) / w[1]

    plt.figure(figsize=(7, 5))

    # Plot decision boundary line
    plt.plot(x_vals, y_vals, "k-", label="Decision Boundary")

    # Plot class 0 and class 1 points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

    plt.title("Logistic Regression Decision Boundary (Dataset 2)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualize decision boundary on training data
plot_decision_boundary(logreg, X_train, y_train)


# ===============================================================
# Task (c): NLL for Different Learning Rates
# ===============================================================

# We will compare how the loss (NLL) behaves for different learning rates.
learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

plt.figure(figsize=(8, 5))

for lr in learning_rates:
    # Create a fresh model for each learning rate
    model = LogisticRegression()

    # Train and collect NLL over iterations
    nll_hist = model.fit(
        X_train,
        y_train,
        lr=lr,
        max_iter=500,
        return_nll=True
    )

    # Plot NLL curve for this learning rate
    plt.plot(nll_hist, label=f"lr={lr}")

plt.xlabel("Iteration")
plt.ylabel("Negative Log-Likelihood (NLL)")
plt.title("NLL over Iterations for Different Learning Rates (Dataset 2)")
plt.legend()
plt.grid(True)
plt.show()
