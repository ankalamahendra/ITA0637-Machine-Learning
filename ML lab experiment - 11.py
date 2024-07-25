import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate a synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature matrix
y = 2 * (X ** 2) + 3 * X + 5 + np.random.randn(100, 1) * 5  # Target vector

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Preprocess the data to include polynomial features
poly = PolynomialFeatures(degree=2)  # Try different degrees for different results
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Step 4: Fit a polynomial regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Step 5: Predict and evaluate the model
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

# Performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Train R2 Score: {train_r2}")
print(f"Test R2 Score: {test_r2}")

# Step 6: Visualize the results
plt.scatter(X, y, color='blue', label='Actual data')
X_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_range_pred = model.predict(poly.transform(X_range))
plt.plot(X_range, y_range_pred, color='red', label='Polynomial regression fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
