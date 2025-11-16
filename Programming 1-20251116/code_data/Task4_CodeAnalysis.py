import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = np.load('dataset3.npz')
X_train = dataset['X_train']
X_test = dataset['X_test']
y_train = dataset['y_train']
y_test = dataset['y_test']


model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("accuracy (test)", accuracy_score(y_test, y_pred))
print("accuracy (train)", accuracy_score(y_train, y_pred_train))
