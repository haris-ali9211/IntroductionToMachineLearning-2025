import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ===============================================================
# Logistic Regression (Numpy Only)
# ===============================================================

class LogisticRegression:

    def __init__(self):
        self.w = None
        self.b = 0.0
        self.nll_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _nll(self, y_true, y_prob):
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    def fit(self, X, y, lr=0.0001, max_iter=1000, return_nll=False):
        n_samples, n_features = X.shape
        y = y.astype(float)

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.nll_history = []

        for i in range(max_iter):
            z = X.dot(self.w) + self.b
            y_hat = self._sigmoid(z)

            nll = self._nll(y, y_hat)
            self.nll_history.append(nll)

            error = (y_hat - y)
            grad_w = X.T.dot(error) / n_samples
            grad_b = np.mean(error)

            self.w -= lr * grad_w
            self.b -= lr * grad_b

        if return_nll:
            return self.nll_history

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ===============================================================
# Load Dataset 2
# ===============================================================

dataset = np.load("dataset2.npz")
X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_train = dataset["y_train"]
y_test = dataset["y_test"]

# ===============================================================
# Task (a): Train Logistic Regression
# ===============================================================

logreg = LogisticRegression()
logreg.fit(X_train, y_train, lr=1e-4, max_iter=1000)

y_pred = logreg.predict(X_test)
print("Task (a) - Test Accuracy:", accuracy_score(y_test, y_pred))


# ===============================================================
# Task (b): Decision Boundary Plot
# ===============================================================

def plot_decision_boundary(model, X, y):
    w = model.w
    b = model.b

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 200)

    # w1*x + w2*y + b = 0  -->  y = -(w1*x + b) / w2
    if abs(w[1]) < 1e-8:  # avoid division by zero
        y_vals = np.zeros_like(x_vals)
    else:
        y_vals = -(w[0] * x_vals + b) / w[1]

    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals, "k-", label="Decision Boundary")

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

    plt.title("Logistic Regression Decision Boundary")
    plt.legend()
    plt.show()


plot_decision_boundary(logreg, X_train, y_train)


# ===============================================================
# Task (c): NLL for Different Learning Rates
# ===============================================================

learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

plt.figure(figsize=(8, 5))
for lr in learning_rates:
    model = LogisticRegression()
    nll_hist = model.fit(X_train, y_train, lr=lr, max_iter=500, return_nll=True)
    plt.plot(nll_hist, label=f"lr={lr}")

plt.xlabel("Iteration")
plt.ylabel("Negative Log-Likelihood (NLL)")
plt.title("NLL over Iterations for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()
