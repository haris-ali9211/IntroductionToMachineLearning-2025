import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------
# Load dataset 3
# ------------------------------
dataset = np.load('dataset3.npz')
X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']

# ------------------------------
# Visualize dataset
# ------------------------------
plt.figure(figsize=(6,5))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("Dataset 3 Visualization")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()

# ------------------------------
# Baseline
# ------------------------------
baseline = max(np.mean(y_train == 0), np.mean(y_train == 1))
print("Baseline accuracy:", baseline)

# ------------------------------
# Logistic Regression (fails)
# ------------------------------
model_lr = LogisticRegression(max_iter=5000)
model_lr.fit(X_train, y_train)

print("\nLogistic Regression:")
print("Train accuracy:", accuracy_score(y_train, model_lr.predict(X_train)))
print("Test accuracy:", accuracy_score(y_test, model_lr.predict(X_test)))

# ------------------------------
# K-NN (works)
# ------------------------------
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)

print("\nKNN:")
print("Train accuracy:", accuracy_score(y_train, model_knn.predict(X_train)))
print("Test accuracy:", accuracy_score(y_test, model_knn.predict(X_test)))
