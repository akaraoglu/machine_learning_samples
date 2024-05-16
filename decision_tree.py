import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# Convert categorical labels to numerical
iris_data['species'] = iris_data['species'].astype('category').cat.codes

# Split the data into features and target
X = iris_data.drop('species', axis=1).values
y = iris_data['species'].values

# Train-test split
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train = X[indices[:train_size]]
y_train = y[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y[indices[train_size:]]

# Decision Tree functions
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples < 1:
            return None

        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, n_samples, n_features)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y, n_samples, n_features):

        if n_samples <= 1:
            return None, None
        
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]

        best_idx, best_thr = None, None
        best_gini = 1.0
        for idx in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_samples_per_class.copy()
            for i in range(1, n_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (n_samples - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (n_samples - i) * gini_right) / n_samples
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _predict(self, inputs):
        node = self.tree
        while 'feature_index' in node:
            if inputs[node['feature_index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

# Train the model
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Classification report
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the decision tree
def plot_decision_tree(node, depth=0):
    if node is None:
        return
    if 'feature_index' in node:
        print(f"{'| ' * depth}├── Feature {node['feature_index']} < {node['threshold']}")
        plot_decision_tree(node['left'], depth + 1)
        plot_decision_tree(node['right'], depth + 1)
    else:
        print(f"{'| ' * depth}├── Leaf: Class {node['predicted_class']}")

plot_decision_tree(clf.tree)
