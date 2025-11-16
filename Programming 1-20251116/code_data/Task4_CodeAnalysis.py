import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# ===============================================================
# Load Dataset 3
# ===============================================================

dataset = np.load("dataset3.npz")
X_train = dataset["X_train"]
X_test = dataset["X_test"]
y_train = dataset["y_train"]
y_test = dataset["y_test"]


# ===============================================================
# Visualize dataset (to understand its structure)
# ===============================================================

plt.figure(figsize=(6, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("Dataset 3 Visualization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# ===============================================================
# Baseline accuracy (majority class)
# ===============================================================

majority = max(np.mean(y_train == 0), np.mean(y_train == 1))
print(f"Baseline (majority class) accuracy: {majority:.3f}")


# ===============================================================
# Train ORIGINAL Logistic Regression (linear boundary)
# ===============================================================

model_linear = LogisticRegression(max_iter=5000)
model_linear.fit(X_train, y_train)

y_pred_test = model_linear.predict(X_test)
y_pred_train = model_linear.predict(X_train)

print("\n--- Original Logistic Regression (Linear) ---")
print("Train accuracy:", accuracy_score(y_train, y_pred_train))
print("Test accuracy :", accuracy_score(y_test, y_pred_test))


# ===============================================================
# Plot linear decision boundary (optional but helpful)
# ===============================================================

def plot_decision_boundary(model, X, y, title):
    w = model.coef_[0]
    b = model.intercept_[0]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 100)

    if abs(w[1]) < 1e-8:
        y_vals = np.zeros_like(x_vals)
    else:
        y_vals = -(w[0] * x_vals + b) / w[1]

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    plt.plot(x_vals, y_vals, "k--", label="Linear Decision Boundary")
    plt.title(title)
    plt.legend()
    plt.show()


plot_decision_boundary(model_linear, X_train, y_train, "Linear Logistic Regression Boundary")


# ===============================================================
# FIX: Add Polynomial Features for Nonlinear Decision Boundary
# ===============================================================

poly = PolynomialFeatures(degree=3)  # degree=3 is enough for complex shapes
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LogisticRegression(max_iter=5000, C=100)  # weaker regularization
model_poly.fit(X_train_poly, y_train)

y_pred_test_poly = model_poly.predict(X_test_poly)
y_pred_train_poly = model_poly.predict(X_train_poly)

print("\n--- Logistic Regression with Polynomial Features (degree=3) ---")
print("Train accuracy:", accuracy_score(y_train, y_pred_train_poly))
print("Test accuracy :", accuracy_score(y_test, y_pred_test_poly))
