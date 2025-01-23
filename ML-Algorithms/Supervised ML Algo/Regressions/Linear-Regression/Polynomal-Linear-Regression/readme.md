# Polynomial Regression with Python

## Introduction
Polynomial Regression is a type of regression analysis where the relationship between the independent variable (X) and the dependent variable (y) is modeled as an nth-degree polynomial. It extends simple linear regression to allow for a nonlinear relationship. This technique is particularly useful when the data shows curvature that cannot be captured by a straight line.

This project demonstrates the implementation of Polynomial Regression using Python libraries, such as NumPy, Matplotlib, Scikit-learn, and Plotly.

---

## What is Polynomial Regression?
Polynomial regression is a method used to model the relationship between variables when the data exhibits a curvilinear pattern. For instance, the general equation of a polynomial regression model of degree 2 is:

**y = b0 + b1*x + b2*x^2 + … + bn*x^n + ε**

Where:
- **y**: Dependent variable
- **x**: Independent variable
- **b0, b1, b2, ..., bn**: Coefficients
- **n**: Degree of the polynomial
- **ε**: Error term

---

## Code Explanation

### Step 1: Importing Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
```
This imports essential libraries:
- **NumPy**: For numerical operations and array manipulations.
- **Matplotlib**: For visualizing data.
- **Scikit-learn**: For machine learning tasks like splitting data, regression models, and evaluating performance.

---

### Step 2: Generating Synthetic Data
```python
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)
```
- **X**: Independent variable generated randomly between -3 and 3.
- **y**: Dependent variable generated using a quadratic equation (0.8\*X^2 + 0.9\*X + 2) with added noise (using `np.random.randn`).

---

### Step 3: Visualizing the Data
```python
plt.plot(X, y, 'b.')
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```
- The data is plotted as a scatter plot to visualize the distribution and observe the curvilinear relationship.

---

### Step 4: Splitting Data into Train and Test Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- **train_test_split**: Splits the dataset into 80% training and 20% testing subsets.
- **random_state**: Ensures reproducibility of results.

---

### Step 5: Applying Linear Regression
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```
- **LinearRegression()**: A linear model is trained on the data.
- **fit()**: Fits the model to the training data.
- **predict()**: Makes predictions on the test data.

#### Evaluate the Model
```python
r2_score(y_test, y_pred)
```
- **r2_score**: Measures how well the predictions fit the actual data (1 is perfect).

---

### Step 6: Applying Polynomial Regression
```python
poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)
```
- **PolynomialFeatures(degree=2)**: Transforms input features into polynomial features (up to degree 2).
- **include_bias=True**: Adds a column of ones to represent the intercept term.

Example:
- Original feature: X = [[2]]
- Transformed: X_trans = [[1, 2, 4]] (1 for bias, 2 for x, and 4 for x^2)

#### Fitting the Polynomial Model
```python
lr.fit(X_train_trans, y_train)
y_pred = lr.predict(X_test_trans)
```
- The transformed features are used to train the model and make predictions.

#### Visualizing Polynomial Regression
```python
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.plot(X_train, y_train, "b.", label='Training points')
plt.plot(X_test, y_test, "g.", label='Testing points')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```
- Polynomial predictions are plotted against the data points.

---

### Step 7: Using Pipelines for Simplified Polynomial Regression
```python
def polynomial_regression(degree):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, 'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(X_train, y_train, "b.", linewidth=3)
    plt.plot(X_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.axis([-3, 3, 0, 10])
    plt.show()
```
- A **Pipeline** is created for sequential transformations and model fitting.
- Parameters include:
  - **poly_features**: Adds polynomial features.
  - **std_scaler**: Standardizes the features for better numerical stability.
  - **lin_reg**: Linear regression model.

---

### Step 8: 3D Polynomial Regression
```python
x = 7 * np.random.rand(100, 1) - 2.8
y = 7 * np.random.rand(100, 1) - 2.8
z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y + 2 + np.random.randn(100, 1)
```
- Generates 3D data with features x, y, and dependent variable z using a quadratic equation with noise.

#### Visualizing 3D Data
```python
import plotly.express as px
fig = px.scatter_3d(x=x.ravel(), y=y.ravel(), z=z.ravel())
fig.show()
```
- A scatter plot visualizes the data in 3D.

#### 3D Polynomial Fitting and Visualization
```python
lr = LinearRegression()
lr.fit(np.array([x, y]).reshape(100, 2), z)
x_input = np.linspace(x.min(), x.max(), 10)
y_input = np.linspace(y.min(), y.max(), 10)
xGrid, yGrid = np.meshgrid(x_input, y_input)
final = np.vstack((xGrid.ravel().reshape(1, 100), yGrid.ravel().reshape(1, 100))).T
z_final = lr.predict(final).reshape(10, 10)
fig = px.scatter_3d(x=x.ravel(), y=y.ravel(), z=z.ravel())
fig.add_trace(go.Surface(x=x_input, y=y_input, z=z_final))
fig.show()
```
- Polynomial fitting in 3D with predictions plotted as a surface.

---

## Key Takeaways
1. **Polynomial Regression** captures non-linear relationships by extending linear regression.
2. **Scikit-learn's PolynomialFeatures** efficiently transforms input features.
3. **Pipelines** simplify complex transformations and scaling workflows.
4. 3D Polynomial Regression and visualization can model and explore multi-dimensional relationships.

