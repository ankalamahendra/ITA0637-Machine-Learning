import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)

X = X[:, np.newaxis]

# Split the data into training/testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1. Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)  # Train the model

# Predict using the linear model
y_pred_linear = linear_regressor.predict(X_test)

# 2. Polynomial Regression (degree 2)
polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X_train)

poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_train)  # Train the model

# Predict using the polynomial model
X_test_poly = polynomial_features.transform(X_test)
y_pred_poly = poly_regressor.predict(X_test_poly)

# Evaluate the models
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Linear Regression MSE: {mse_linear}, R2: {r2_linear}")
print(f"Polynomial Regression MSE: {mse_poly}, R2: {r2_poly}")

# Visualize the results
plt.figure(figsize=(14, 5))

# Plot Linear Regression results
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='black', label='Data')
plt.plot(X_test, y_pred_linear, color='blue', linewidth=2, label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

# Plot Polynomial Regression results
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='black', label='Data')
# Sort the values of X_test for a smoother curve
sorted_X_test = np.sort(X_test, axis=0)
sorted_y_pred_poly = poly_regressor.predict(polynomial_features.transform(sorted_X_test))
plt.plot(sorted_X_test, sorted_y_pred_poly, color='red', linewidth=2, label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (degree=2)')
plt.legend()

plt.show()
