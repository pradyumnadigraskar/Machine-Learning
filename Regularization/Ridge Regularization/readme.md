# Regularization in Machine Learning: L1 and Ridge Regularization

## Introduction
Regularization is a fundamental technique in machine learning that helps prevent overfitting by adding a penalty term to the loss function during training. This penalty term discourages the model from excessively fitting the training data, ensuring better generalization to unseen data.

In this project, we demonstrate L1 and Ridge Regularization with Python using scikit-learn. The code includes examples of how to apply these techniques to linear regression tasks, with detailed explanations of the parameters, methods, and evaluation metrics.

---

## What is Ridge Regularization?
Ridge regularization, also known as **L2 regularization**, adds a penalty proportional to the sum of the squared values of the coefficients. This penalty discourages large coefficients, which can make the model overly sensitive to noise in the training data. The objective function for Ridge regression is:

\[
J(\theta) = \text{MSE} + \alpha \sum_{j=1}^n \theta_j^2
\]

Where:
- **MSE**: Mean Squared Error, the loss function for linear regression.
- **\(\alpha\)**: Regularization strength (penalty term). Higher values mean more regularization.
- **\(\theta_j\)**: Model coefficients.

By minimizing the sum of squared errors and the squared coefficients, Ridge regression prevents overfitting and improves the stability of the model.

---

## Code Explanation

### Step 1: Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
```
**Explanation**:
- **NumPy**: For numerical operations and array manipulations.
- **Pandas**: To handle tabular data (optional in this code).
- **Matplotlib**: For visualizing data.
- **load_diabetes**: A built-in dataset in scikit-learn for regression tasks, containing 10 features and a target variable representing disease progression.

---

### Step 2: Load the Dataset
```python
data = load_diabetes()
print(data.DESCR)
X = data.data
y = data.target
```
**Explanation**:
- `load_diabetes()`: Loads the diabetes dataset, which is a regression dataset.
- `data.DESCR`: Prints the dataset description, explaining the features and target variable.
- `X`: Contains 10 input features (independent variables).
- `y`: Target variable (dependent variable) representing disease progression.

---

### Step 3: Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
```
**Explanation**:
- `train_test_split`: Splits the dataset into training and testing sets.
  - **`test_size=0.2`**: Allocates 20% of the data for testing.
  - **`random_state=45`**: Ensures reproducibility of the split.

---

### Step 4: Fit a Linear Regression Model
```python
from sklearn.linear_model import LinearRegression
L = LinearRegression()
L.fit(X_train, y_train)
```
**Explanation**:
- `LinearRegression`: Initializes a standard linear regression model.
- `L.fit(X_train, y_train)`: Fits the model to the training data by minimizing the Mean Squared Error (MSE).

#### View Model Coefficients and Intercept
```python
print(L.coef_)
print(L.intercept_)
```
**Explanation**:
- **`L.coef_`**: Returns the coefficients (weights) of the model for each feature.
- **`L.intercept_`**: Returns the intercept term (\(b_0\)).

---

### Step 5: Evaluate Linear Regression
```python
y_pred = L.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error

print("R2 score", r2_score(y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
```
**Explanation**:
- **`y_pred`**: Predictions made by the model on the test set.
- **`r2_score`**: Measures how well the model explains the variance in the target variable (1 is perfect).
- **`mean_squared_error`**: Calculates the average squared error between predictions and actual values.
- **`np.sqrt(mean_squared_error)`**: Computes the Root Mean Squared Error (RMSE), making the error interpretable in the same units as the target variable.

---

### Step 6: Ridge Regularization
```python
from sklearn.linear_model import Ridge
R = Ridge(alpha=100000)
R.fit(X_train, y_train)
```
**Explanation**:
- **`Ridge`**: Initializes Ridge regression with an \(\alpha\) value of 100,000.
- **`alpha`**: Controls the regularization strength. A higher \(\alpha\) results in stronger regularization, shrinking the coefficients more aggressively.
- **`R.fit`**: Fits the Ridge regression model to the training data.

#### View Ridge Coefficients and Intercept
```python
print(R.coef_)
print(R.intercept_)
```
**Explanation**:
- **`R.coef_`**: Returns the shrunken coefficients after applying L2 regularization.
- **`R.intercept_`**: Returns the intercept term.

#### Evaluate Ridge Regression
```python
y_pred1 = R.predict(X_test)
print("R2 score", r2_score(y_test, y_pred1))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred1)))
```
**Explanation**:
- **`y_pred1`**: Predictions made by the Ridge regression model.
- **`r2_score`** and **`RMSE`**: Evaluate the Ridge regression model using the same metrics as linear regression.

---

### Step 7: Generate Synthetic Data
```python
m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)
plt.show()
```
**Explanation**:
- `x1`: Independent variable sampled uniformly in the range [-2, 3].
- `x2`: Quadratic relationship with added noise (random Gaussian noise).
- `plt.scatter`: Plots a scatter plot of `x1` and `x2`.

---

### Step 8: Ridge Regression with Polynomial Features
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)
```
**Explanation**:
- **Pipeline**: Sequentially applies transformations and fits the model.
  - **`PolynomialFeatures(degree=16)`**: Transforms `x1` into polynomial features up to degree 16.
  - **`Ridge(alpha=alpha)`**: Applies Ridge regression with the specified \(\alpha\) value.
- **`get_preds_ridge`**: Fits the pipeline and returns predictions.

---

### Step 9: Visualize Ridge Regression with Different \(\alpha\) Values
```python
alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Datapoints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()
```
**Explanation**:
- **`alphas`**: List of \(\alpha\) values to test different regularization strengths.
- **`plt.plot`**: Plots the predictions for each \(\alpha\) value using different colors (`cs`).
- **`np.argsort`**: Sorts the data points for smooth visualization.

---

## Key Takeaways
1. **Ridge Regularization** prevents overfitting by shrinking coefficients towards zero, controlled by the \(\alpha\) parameter.
2. **Pipeline** simplifies the process of applying transformations and fitting models.
3. Increasing \(\alpha\) reduces overfitting but may increase bias.

