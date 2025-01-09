# README.md

## Power Transformer Overview

The `PowerTransformer` is a tool in machine learning preprocessing that applies a power transformation to the features to make data more Gaussian-like. It stabilizes variance, minimizes skewness, and can improve the performance of regression models and algorithms sensitive to data distribution. 

### Types of Power Transformations

1. **Box-Cox Transformation**
   - Suitable for positive data only.
   - Applies a logarithmic transformation if lambda is zero.

2. **Yeo-Johnson Transformation**
   - Handles both positive and negative data.
   - Generalizes the Box-Cox transformation.

## Code Explanation

### Importing Required Libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
```
- **`numpy`**: Numerical operations.
- **`pandas`**: Data manipulation.
- **`seaborn`** and **`matplotlib`**: Data visualization.
- **`scipy.stats`**: Statistical functions.
- **`sklearn` modules**: Model training, evaluation, and transformations.

### Loading and Exploring the Data
```python
df = pd.read_csv('concrete_data.csv')
df.head()
df.shape
df.isnull().sum()
df.describe()
```
- **`pd.read_csv`**: Reads the dataset.
- **`head`, `shape`, `isnull`, `describe`**: Display dataset info, check shape, null values, and summary stats.

### Splitting Features and Target Variable
```python
X = df.drop(columns=['Strength'])
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **`drop`**: Drops the target column (`Strength`) from features.
- **`train_test_split`**: Splits data into training and testing sets. Parameters:
  - `test_size=0.2`: 20% for testing.
  - `random_state=42`: Ensures reproducibility.

### Regression Without Transformation
```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))
```
- **`LinearRegression`**: Initializes the linear regression model.
- **`fit`**: Trains the model on training data.
- **`predict`**: Predicts on test data.
- **`r2_score`**: Evaluates model performance (R²).

### Cross-Validation Without Transformation
```python
lr = LinearRegression()
np.mean(cross_val_score(lr, X, y, scoring='r2'))
```
- **`cross_val_score`**: Performs cross-validation. Parameters:
  - `scoring='r2'`: Uses R² for evaluation.

### Visualizing Data Distribution
```python
for col in X_train.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.title(col)
    plt.show()
```
- Loops through each column to:
  - Plot distributions with **`sns.distplot`**.
  - Generate QQ plots with **`stats.probplot`**.

### Box-Cox Transformation
```python
pt = PowerTransformer(method='box-cox')
X_train_transformed = pt.fit_transform(X_train + 0.000001)
X_test_transformed = pt.transform(X_test + 0.000001)
pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_})
```
- **`PowerTransformer(method='box-cox')`**: Applies Box-Cox transformation. Adds a small constant to handle zero values.
- **`fit_transform`**: Fits and transforms training data.
- **`lambdas_`**: Stores the lambda values for each feature.

### Regression with Box-Cox Transformation
```python
lr = LinearRegression()
lr.fit(X_train_transformed, y_train)
y_pred2 = lr.predict(X_test_transformed)
r2_score(y_test, y_pred2)
```
- Similar steps as before, using transformed data.

### Cross-Validation with Box-Cox
```python
X_transformed = pt.fit_transform(X + 0.0000001)
np.mean(cross_val_score(lr, X_transformed, y, scoring='r2'))
```
- Applies cross-validation on transformed features.

### Visualizing Box-Cox Transformation
```python
for col in X_train_transformed.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(X_train_transformed[col])
    plt.title(col)
    plt.show()
```
- Compares distributions before and after Box-Cox transformation.

### Yeo-Johnson Transformation
```python
pt1 = PowerTransformer()
X_train_transformed2 = pt1.fit_transform(X_train)
X_test_transformed2 = pt1.transform(X_test)
lr.fit(X_train_transformed2, y_train)
y_pred3 = lr.predict(X_test_transformed2)
print(r2_score(y_test, y_pred3))
```
- **`PowerTransformer()`**: Defaults to Yeo-Johnson.
- Suitable for data with negative values.

### Cross-Validation with Yeo-Johnson
```python
pt = PowerTransformer()
X_transformed2 = pt.fit_transform(X)
np.mean(cross_val_score(lr, X_transformed2, y, scoring='r2'))
```
- Performs cross-validation using Yeo-Johnson transformed data.

### Visualizing Yeo-Johnson Transformation
```python
for col in X_train_transformed2.columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)

    plt.subplot(122)
    sns.distplot(X_train_transformed2[col])
    plt.title(col)
    plt.show()
```
- Compares distributions before and after Yeo-Johnson transformation.

### Comparing Box-Cox and Yeo-Johnson Lambdas
```python
pd.DataFrame({'cols': X_train.columns, 'box_cox_lambdas': pt.lambdas_, 'Yeo_Johnson_lambdas': pt1.lambdas_})
```
- Compares the lambda values for each transformation.

## Summary
This script demonstrates the importance of data transformation in regression tasks. Both Box-Cox and Yeo-Johnson transformations improve the normality of features, reducing skewness and stabilizing variance. The comparison between lambdas highlights differences in the transformations.

