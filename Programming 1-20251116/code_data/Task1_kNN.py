import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================================
# Load dataset1
# ================================
dataset = np.load('dataset1.npz')
X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']

# =====================================
# Task 1(a): Train k-NN with k = 5
# =====================================
print("\n================ TASK 1(a) ================")

k5 = KNeighborsClassifier(n_neighbors=5)
k5.fit(X_train, y_train)

train_pred_k5 = k5.predict(X_train)
test_pred_k5 = k5.predict(X_test)

print("Train accuracy (k=5):", accuracy_score(y_train, train_pred_k5))
print("Test accuracy  (k=5):", accuracy_score(y_test, test_pred_k5))
print("Observation: Train accuracy is slightly higher than test accuracy, which is expected.\n")


# =====================================
# Task 1(b): Find best k using validation
# =====================================
print("================ TASK 1(b) ================")

# Split training into train+validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

k_values = range(1, 21)   # try k = 1 to k = 20
val_accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    val_accuracies.append(acc)

# Plot validation accuracy vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, val_accuracies, marker='o')
plt.xlabel("k (number of neighbors)")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy for Different k")
plt.grid(True)
plt.show()

# Select the best k
best_k = k_values[np.argmax(val_accuracies)]
print("Best k found from validation:", best_k)

# Train final model with full training data
kNN_final = KNeighborsClassifier(n_neighbors=best_k)
kNN_final.fit(X_train, y_train)

final_test_pred = kNN_final.predict(X_test)
print("Final Test Accuracy with best k:", accuracy_score(y_test, final_test_pred))
print("Observation: The test accuracy with best k may or may not improve depending on dataset variance.\n")

# =====================================
# Task 1(c): Can we reuse the same k for other datasets?
# =====================================
print("================ TASK 1(c) ================")
print("Answer:")
print("You cannot reuse the same k for another dataset because the optimal k depends on the dataset's noise,")
print("density, distribution of classes, and separation between clusters. Each new dataset requires tuning k again.")
