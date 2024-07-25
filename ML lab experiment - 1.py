# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
np.random.seed(42)

# Define the number of samples
num_samples = 1000

# Create features
battery_power = np.random.randint(1000, 5000, num_samples)  # Increase the range to make it more predictive
blue = np.random.randint(0, 2, num_samples)
clock_speed = np.random.uniform(1.0, 3.0, num_samples)  # Narrow the range to make it more predictive
dual_sim = np.random.randint(0, 2, num_samples)
fc = np.random.randint(0, 20, num_samples)
four_g = np.random.randint(0, 2, num_samples)
int_memory = np.random.randint(2, 128, num_samples)
m_dep = np.random.uniform(0.1, 1.0, num_samples)
mobile_wt = np.random.randint(80, 250, num_samples)
n_cores = np.random.randint(1, 8, num_samples)
pc = np.random.randint(0, 20, num_samples)
px_height = np.random.randint(0, 1960, num_samples)
px_width = np.random.randint(0, 2000, num_samples)
ram = np.random.randint(256, 8000, num_samples)  # Increase the range to make it more predictive
sc_h = np.random.randint(5, 19, num_samples)
sc_w = np.random.randint(0, 18, num_samples)
talk_time = np.random.randint(2, 20, num_samples)
three_g = np.random.randint(0, 2, num_samples)
touch_screen = np.random.randint(0, 2, num_samples)
wifi = np.random.randint(0, 2, num_samples)

# Create target variable: price_range (0: low, 1: medium, 2: high, 3: very high)
price_range = np.select(
    [ram < 2000, ram < 4000, ram < 6000],
    [0, 1, 2],
    default=3
)

# Create DataFrame
data = pd.DataFrame({
    'battery_power': battery_power,
    'blue': blue,
    'clock_speed': clock_speed,
    'dual_sim': dual_sim,
    'fc': fc,
    'four_g': four_g,
    'int_memory': int_memory,
    'm_dep': m_dep,
    'mobile_wt': mobile_wt,
    'n_cores': n_cores,
    'pc': pc,
    'px_height': px_height,
    'px_width': px_width,
    'ram': ram,
    'sc_h': sc_h,
    'sc_w': sc_w,
    'talk_time': talk_time,
    'three_g': three_g,
    'touch_screen': touch_screen,
    'wifi': wifi,
    'price_range': price_range
})

# Step b: Print the first five rows
print(data.head())

# Step c: Basic statistical computations on the dataset
print(data.describe())

# Step d: Display columns and their data types
print(data.dtypes)

# Step e: Detect and handle null values (there should be none in this synthetic dataset)
print(data.isnull().sum())

# Step f: Explore the dataset using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step g: Split the data into training and testing sets
X = data.drop('price_range', axis=1)  # Features
y = data['price_range']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step h: Fit the model using Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step i: Predict the model
y_pred = model.predict(X_test)

# Step j: Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')
