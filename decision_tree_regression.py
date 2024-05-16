import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree Regression
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_split = self._best_split(X, y, n_samples, n_features)
            if best_split:
                left_tree = self._grow_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
                right_tree = self._grow_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
                return {'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'], 'left': left_tree, 'right': right_tree}

        return {'value': np.mean(y)}

    def _best_split(self, X, y, n_samples, n_features):
        best_split = {}
        best_mse = float('inf')
        for feature_index in range(n_features):
            thresholds, mse_values = self._find_thresholds(X[:, feature_index], y)
            if thresholds.any():
                min_mse_index = np.argmin(mse_values)
                if mse_values[min_mse_index] < best_mse:
                    best_mse = mse_values[min_mse_index]
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': thresholds[min_mse_index],
                        'left_indices': np.where(X[:, feature_index] <= thresholds[min_mse_index])[0],
                        'right_indices': np.where(X[:, feature_index] > thresholds[min_mse_index])[0]
                    }
        return best_split

    def _find_thresholds(self, feature_column, y):
        thresholds = np.unique(feature_column)
        if len(thresholds) < 2:
            return None, None

        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        mse_values = []
        for threshold in thresholds:
            left_indices = np.where(feature_column <= threshold)[0]
            right_indices = np.where(feature_column > threshold)[0]
            mse = self._calculate_mse(y[left_indices], y[right_indices])
            mse_values.append(mse)
        return thresholds, mse_values

    def _calculate_mse(self, left_y, right_y):
        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf')

        left_mse = np.mean((left_y - np.mean(left_y)) ** 2)
        right_mse = np.mean((right_y - np.mean(right_y)) ** 2)
        return len(left_y) * left_mse + len(right_y) * right_mse

    def _predict(self, inputs, tree):
        if 'value' in tree:
            return tree['value']
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if inputs[feature_index] <= threshold:
            return self._predict(inputs, tree['left'])
        else:
            return self._predict(inputs, tree['right'])

# Train the model
dt_regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred = dt_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Plotting the error distribution
errors = y_pred - y_test
plt.hist(errors, bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.show()
