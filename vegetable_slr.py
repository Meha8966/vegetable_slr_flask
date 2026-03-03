import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset (Make sure Vegetable_market.csv is in same folder)
dataset = pd.read_csv("Vegetable_market.csv")

# Select feature and target
X = dataset.iloc[:, 3:4].values
y = dataset.iloc[:, 6:7].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test values
y_pred = regressor.predict(X_test)

# Print model score
print("Model Score:", regressor.score(X, y))

# Sample prediction
predo = regressor.predict([[2]])
print("Prediction for input 2:", predo)

# First Graph
plt.scatter(X_train[:, 0], y_train, color="orange")
plt.plot(X_train[:, 0], regressor.predict(X_train), color="blue")

plt.title("Vegetable Market Price Prediction")
plt.xlabel("Feature 1")
plt.ylabel("Price")

plt.show()

# Second Graph (Smooth line)
x_feature = X_train[:, 0]
x_range = np.linspace(x_feature.min(), x_feature.max(), 100).reshape(-1, 1)

X_range_full = np.zeros((100, X_train.shape[1]))
X_range_full[:, 0] = x_range[:, 0]

y_pred_range = regressor.predict(X_range_full)

plt.figure()
plt.scatter(x_feature, y_train)
plt.plot(x_range, y_pred_range)

plt.xlabel("Feature 1")
plt.ylabel("Price")
plt.title("Vegetable Market Linear Regression")

plt.show()


plt.show()

import pickle
pickle.dump(regressor, open("model.pkl", "wb"))