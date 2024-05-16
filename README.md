# Machine Learning Algorithms Quick Review

![1_rbaxTrB_CZCqbty_zv2bEg-3005608504](https://github.com/akaraoglu/machine_learning_samples/assets/32932292/c8520976-d2d7-47c3-86a9-fe332c204dbe)

## Naive Bayes Classifier: 
A probabilistic classifier based on Bayes' theorem, assuming feature independence. It's simple and effective for text classification.
Advantages: Simple, fast, and works well with high-dimensional data.
Disadvantages: Assumes feature independence, which may not always be true.

## Decision Trees: 
Models decisions using a tree structure, splitting data based on feature values. Useful for both classification and regression.
Advantages: Easy to interpret and visualize, handles both categorical and numerical data.
Disadvantages: Prone to overfitting, sensitive to small changes in data.

## Support Vector Machines (SVM): 
Finds the optimal hyperplane to separate classes in a high-dimensional space. Effective in high-dimensional spaces.
Advantages: Effective in high-dimensional spaces, works well with clear margin of separation.
Disadvantages: Computationally intensive, less effective with large datasets and noisy data.

## Random Forest: 
An ensemble method that combines multiple decision trees to enhance predictive accuracy. Reduces overfitting by averaging multiple trees.
Advantages: Reduces overfitting, robust to noise, handles missing values well.
Disadvantages: Less interpretable, can be slow with a large number of trees.

## K-Nearest Neighbors (KNN): 
Classifies data points by the majority class among their k-nearest neighbors. Simple and intuitive but can be computationally expensive.
Advantages: Simple and intuitive, no training phase.
Disadvantages: Computationally expensive at prediction time, sensitive to irrelevant features.

## Linear Regression: 
Models the linear relationship between input variables and the output. Used for predicting continuous outcomes.
Advantages: Simple, easy to interpret, fast to train.
Disadvantages: Assumes linear relationship, sensitive to outliers.

## Neural Network Regression: 
Uses layers of neurons to capture complex relationships in data for regression tasks. Suitable for non-linear problems.
Advantages: Captures complex, non-linear relationships, highly flexible.
Disadvantages: Requires large amounts of data, computationally expensive.

## Support Vector Regression (SVR): 
Extends SVM for regression, aiming to fit the best line within a margin of error. Handles high-dimensional spaces well.
Advantages: Effective in high-dimensional spaces, robust to outliers.
Disadvantages: Computationally intensive, difficult to tune hyperparameters.

## Decision Tree Regression: 
Uses a tree structure to predict continuous outcomes by splitting data based on feature values. Easy to interpret.
Advantages: Easy to interpret, handles both categorical and numerical data.
Disadvantages: Prone to overfitting, sensitive to small changes in data.

## Lasso Regression: 
A linear regression with L1 regularization to enforce sparsity, reducing the number of features. Helps prevent overfitting.
Advantages: Performs feature selection, prevents overfitting.
Disadvantages: Can exclude important features, not suitable for highly correlated features.

## Ridge Regression: 
Similar to linear regression but includes L2 regularization to penalize large coefficients, improving generalization.
Advantages: Reduces overfitting, handles multicollinearity.
Disadvantages: Coefficients are harder to interpret, does not perform feature selection.

## K-Means Clustering: 
Partitions data into k clusters by minimizing variance within each cluster. Efficient for large datasets.
Advantages: Simple, efficient for large datasets.
Disadvantages: Requires specifying number of clusters, sensitive to initial placement.

## Mean-shift Clustering: 
Identifies clusters by locating areas of high data point density. Does not require specifying the number of clusters.
Advantages: Does not require specifying number of clusters, can find arbitrarily shaped clusters.
Disadvantages: Computationally intensive, choice of bandwidth parameter is crucial.

## DBSCAN Clustering:
Groups points based on density, identifying clusters and noise. Works well with arbitrarily shaped clusters.
Advantages: Identifies clusters of arbitrary shape, robust to noise.
Disadvantages: Struggles with varying density, sensitive to parameter selection.

## Agglomerative Hierarchical Clustering: 
Builds clusters hierarchically by merging smaller clusters into larger ones. Useful for understanding data structure.
Advantages: Does not require specifying number of clusters, provides a dendrogram for visualization.
Disadvantages: Computationally expensive, less effective with large datasets.

## Gaussian Mixture: 
Models data as a mixture of several Gaussian distributions, estimating the probability of each point belonging to a cluster. Flexible and powerful for complex distributions.
Advantages: Can model complex distributions, flexible.
Disadvantages: Requires specifying number of components, sensitive to initialization.

## Q-Learning: 
A reinforcement learning algorithm that learns the value of actions to maximize cumulative reward. Does not require a model of the environment.
Advantages: Model-free, can handle stochastic environments.
Disadvantages: Can be slow to converge, requires a lot of exploration.

## R Learning: 
Focuses on maximizing the average reward per time step in reinforcement learning tasks. Useful for ongoing, long-term tasks.
Advantages: Maximizes long-term average reward, suitable for ongoing tasks.
Disadvantages: Complex to implement, requires a lot of data.

## TD Learning (Temporal Difference Learning): 
Combines ideas from Monte Carlo and dynamic programming to predict future rewards. Learns directly from raw experience.
Advantages: Learns directly from raw experience, efficient.
Disadvantages: Requires careful tuning of learning rate, can be unstable.
