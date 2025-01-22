# Multiple Linear Regression

## Overview
Multiple Linear Regression is a statistical method used to model the relationship between one dependent variable (target) and two or more independent variables (features). It assumes a linear relationship between the dependent and independent variables and aims to predict the target variable based on the values of the features.

In this example, we will:
1. Use the `Diabetes` dataset from Scikit-learn to predict a target variable based on several features.
2. Implement and evaluate a multiple linear regression model using Scikit-learn.
3. Implement our own Linear Regression class (`MeraLR`) from scratch using NumPy.

---

## Code Explanation

### 1. Importing Required Libraries
```python
import numpy as np
from sklearn.datasets import load_diabetes
```
- **`numpy`**: A library for numerical computations in Python.
- **`sklearn.datasets`**: Provides access to the `load_diabetes` dataset, which contains data about diabetes patients, including features like age, BMI, and others.

---

### 2. Loading the Dataset
```python
X, y = load_diabetes(return_X_y=True)
```
- **`load_diabetes(return_X_y=True)`**: Loads the dataset and returns two components:
  - `X`: Feature matrix with shape `(442, 10)` where 442 is the number of samples, and 10 is the number of features.
  - `y`: Target variable with shape `(442,)`, representing a continuous value for each sample.

```python
X.shape
```
- Outputs the shape of the feature matrix `X`.

```python
y.shape
```
- Outputs the shape of the target variable `y`.

---

### 3. Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- **`train_test_split`**: Splits the dataset into training and testing sets.
  - `X_train`: 80% of the feature matrix for training.
  - `X_test`: 20% of the feature matrix for testing.
  - `y_train`: Corresponding target values for training.
  - `y_test`: Corresponding target values for testing.
  - **`test_size=0.2`**: Specifies the test set size as 20%.
  - **`random_state=2`**: Ensures reproducibility of the split.

```python
print(X_train.shape)
print(X_test.shape)
```
- Outputs the shapes of `X_train` and `X_test`.

---

### 4. Implementing Linear Regression using Scikit-learn
```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
```
- **`LinearRegression()`**: Creates a linear regression model.
- **`fit(X_train, y_train)`**: Trains the model by finding the best-fit coefficients (`coef_`) and intercept (`intercept_`) to minimize the prediction error on the training data.

```python
y_pred = reg.predict(X_test)
```
- **`predict(X_test)`**: Generates predictions for the test data using the trained model.

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```
- **`r2_score(y_test, y_pred)`**: Calculates the coefficient of determination (R² score), which measures how well the model explains the variance in the target variable.

```python
reg.coef_
```
- Outputs the coefficients for each feature.

```python
reg.intercept_
```
- Outputs the intercept term.

---

### 5. Creating a Custom Linear Regression Class (`MeraLR`)
```python
class MeraLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
```
- **`__init__`**: Initializes the class.
  - `coef_`: Stores the coefficients for the features.
  - `intercept_`: Stores the intercept term.

```python
    def fit(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
```
- **`fit`**: Trains the custom linear regression model.
  - `np.insert(X_train, 0, 1, axis=1)`: Adds a column of ones to the feature matrix for the intercept term.
  - **`np.dot(X_train.T, X_train)`**: Computes the dot product of the transpose of `X_train` and `X_train`.
  - **`np.linalg.inv()`**: Computes the inverse of the matrix.
  - **`.dot(X_train.T).dot(y_train)`**: Solves the normal equation to calculate the coefficients.

```python
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        return y_pred
```
- **`predict`**: Generates predictions by calculating the dot product of the test features and coefficients, then adding the intercept.

---

### 6. Training and Evaluating `MeraLR`
```python
lr = MeraLR()
lr.fit(X_train, y_train)
```
- **`lr.fit(X_train, y_train)`**: Trains the custom model using the training data.

```python
X_train.shape
np.insert(X_train, 0, 1, axis=1).shape
```
- Confirms the shape of the feature matrix after adding the intercept column.

```python
y_pred = lr.predict(X_test)
r2_score(y_test, y_pred)
```
- Generates predictions and evaluates the R² score for the custom model.

```python
lr.coef_
lr.intercept_
```
- Outputs the coefficients and intercept of the custom model.

---

## Key Concepts in Multiple Linear Regression
1. **Features and Target**:
   - **Features**: Independent variables used to predict the target.
   - **Target**: Dependent variable (outcome).

2. **Coefficients and Intercept**:
   - **Coefficients**: Measure the contribution of each feature to the target.
   - **Intercept**: Value of the target when all features are zero.

3. **R² Score**:
   - Indicates how well the model fits the data.
   - Value ranges from 0 to 1, where higher values indicate better performance.

---

## Conclusion
This project demonstrates multiple linear regression using Scikit-learn and a custom implementation with NumPy. It highlights the importance of understanding mathematical concepts like the normal equation and matrix operations to build and evaluate regression models effectively.

