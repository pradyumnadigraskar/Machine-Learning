# Random Sample Imputation with Python

## Overview
Random Sample Imputation is a technique used to handle missing data in datasets. Instead of imputing missing values with mean, median, or mode, this method randomly selects a value from the non-missing data for the same feature to replace the missing values. It helps preserve the original distribution and variability of the data, which is crucial for some machine learning models.

This README explains Random Sample Imputation, provides step-by-step commentary on the code, and discusses the parameters used.

---

## What is Random Sample Imputation?
Random Sample Imputation works as follows:
1. Identify missing values in a dataset.
2. For each missing value, randomly select an observed value from the same column.
3. Replace the missing value with the randomly selected value.

### Advantages:
- Preserves the original distribution and variability of the data.
- Suitable for numerical and categorical variables.

### Disadvantages:
- Introduces randomness, which may affect reproducibility if not controlled.
- Not suitable for small datasets with limited non-missing values.

---

## Code Explanation

### Importing Required Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
```
- **`numpy` and `pandas`**: Libraries for numerical computations and data manipulation.
- **`train_test_split`**: Splits the dataset into training and testing sets.
- **`matplotlib` and `seaborn`**: Libraries for data visualization.

### Data Loading and Initial Exploration
```python
df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
df.head()
df.isnull().mean() * 100
```
- **`pd.read_csv`**: Loads the Titanic dataset.
- **`usecols`**: Specifies columns `Age`, `Fare`, and `Survived` for analysis.
- **`isnull().mean() * 100`**: Calculates the percentage of missing values in each column.

### Splitting the Data
```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- **`X`**: Features (`Age`, `Fare`).
- **`y`**: Target variable (`Survived`).
- **`train_test_split`**: Splits the data into training and testing sets with 20% for testing.

### Random Sample Imputation for Numeric Data
```python
X_train['Age_imputed'] = X_train['Age']
X_test['Age_imputed'] = X_test['Age']
```
- Creates new columns `Age_imputed` in training and testing data to store imputed values.

```python
X_train['Age_imputed'][X_train['Age_imputed'].isnull()] = X_train['Age'].dropna().sample(X_train['Age'].isnull().sum()).values
X_test['Age_imputed'][X_test['Age_imputed'].isnull()] = X_train['Age'].dropna().sample(X_test['Age'].isnull().sum()).values
```
- **`isnull()`**: Identifies missing values.
- **`dropna()`**: Removes missing values from `Age`.
- **`sample()`**: Randomly selects values for imputation.
- **`values`**: Converts sampled values into a format suitable for assignment.

### Distribution Analysis
```python
sns.distplot(X_train['Age'], label='Original', hist=False)
sns.distplot(X_train['Age_imputed'], label='Imputed', hist=False)
plt.legend()
plt.show()
```
- **`sns.distplot`**: Plots the distribution of `Age` before and after imputation.
- **`plt.legend()`**: Adds a legend to distinguish original and imputed data.

### Variance Analysis
```python
print('Original variable variance: ', X_train['Age'].var())
print('Variance after random imputation: ', X_train['Age_imputed'].var())
```
- Compares the variance of `Age` before and after imputation.

### Covariance and Boxplot Analysis
```python
X_train[['Fare', 'Age', 'Age_imputed']].cov()
X_train[['Age', 'Age_imputed']].boxplot()
```
- **`cov()`**: Computes covariance between variables.
- **`boxplot()`**: Displays the range, quartiles, and outliers.

### Random Sample Imputation for Categorical Data
```python
data = pd.read_csv('house-train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
data.isnull().mean() * 100
X = data
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- **`GarageQual`** and **`FireplaceQu`**: Categorical features with missing values.

```python
X_train['GarageQual_imputed'] = X_train['GarageQual']
X_test['GarageQual_imputed'] = X_test['GarageQual']
X_train['FireplaceQu_imputed'] = X_train['FireplaceQu']
X_test['FireplaceQu_imputed'] = X_test['FireplaceQu']
```
- Creates imputed columns for categorical features.

```python
X_train['GarageQual_imputed'][X_train['GarageQual_imputed'].isnull()] = X_train['GarageQual'].dropna().sample(X_train['GarageQual'].isnull().sum()).values
X_test['GarageQual_imputed'][X_test['GarageQual_imputed'].isnull()] = X_train['GarageQual'].dropna().sample(X_test['GarageQual'].isnull().sum()).values
```
- Performs random sample imputation for `GarageQual`.

```python
X_train['FireplaceQu_imputed'][X_train['FireplaceQu_imputed'].isnull()] = X_train['FireplaceQu'].dropna().sample(X_train['FireplaceQu'].isnull().sum()).values
X_test['FireplaceQu_imputed'][X_test['FireplaceQu_imputed'].isnull()] = X_train['FireplaceQu'].dropna().sample(X_test['FireplaceQu'].isnull().sum()).values
```
- Performs random sample imputation for `FireplaceQu`.

### Distribution Comparison for Categorical Data
```python
temp = pd.concat([
    X_train['GarageQual'].value_counts() / len(X_train['GarageQual'].dropna()),
    X_train['GarageQual_imputed'].value_counts() / len(X_train)], axis=1)
temp.columns = ['original', 'imputed']
temp
```
- Compares the distribution of original and imputed `GarageQual` values.

```python
for category in X_train['FireplaceQu'].dropna().unique():
    sns.distplot(X_train[X_train['FireplaceQu'] == category]['SalePrice'], hist=False, label=category)
plt.show()
```
- Visualizes the relationship between `FireplaceQu` categories and `SalePrice`.

```python
for category in X_train['FireplaceQu_imputed'].dropna().unique():
    sns.distplot(X_train[X_train['FireplaceQu_imputed'] == category]['SalePrice'], hist=False, label=category)
plt.show()
```
- Visualizes the imputed categorical feature and its relationship with `SalePrice`.

---

## Conclusion
Random Sample Imputation is a robust method for handling missing data while preserving original distributions. By ensuring variability and integrity of the data, it can improve the performance of machine learning models.

