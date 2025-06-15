# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:57:12 2025

@author: USER PC
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Generate Synthetic Linear Data
# ----------------------------
np.random.seed(42)
X = 2 * np.random.rand(100, 1)                # Feature values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)       # Linear relation with noise

# ----------------------------
# 2. Train Linear Regression (OLS)
# ----------------------------
ols_model = LinearRegression()
ols_model.fit(X, y)
y_pred_ols = ols_model.predict(X)

# ----------------------------
# 3. Train Stochastic Gradient Descent (SGD)
# ----------------------------
sgd_model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant', random_state=42)
sgd_model.fit(X, y.ravel())
y_pred_sgd = sgd_model.predict(X)

# ----------------------------
# 4. Print Model Performance
# ----------------------------
print("=== Ordinary Least Squares (OLS) ===")
print(f"Intercept     : {ols_model.intercept_[0]:.4f}")
print(f"Coefficient   : {ols_model.coef_[0][0]:.4f}")
print(f"MSE           : {mean_squared_error(y, y_pred_ols):.4f}\n")

print("=== Stochastic Gradient Descent (SGD) ===")
print(f"Intercept     : {sgd_model.intercept_[0]:.4f}")
print(f"Coefficient   : {sgd_model.coef_[0]:.4f}")
print(f"MSE           : {mean_squared_error(y, y_pred_sgd):.4f}")

# ----------------------------
# 5. Visualization
# ----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.6, label='Data')
plt.plot(X, y_pred_ols, color='blue', label='OLS Prediction')
plt.plot(X, y_pred_sgd, color='red', linestyle='--', label='SGD Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison: OLS vs SGD Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
