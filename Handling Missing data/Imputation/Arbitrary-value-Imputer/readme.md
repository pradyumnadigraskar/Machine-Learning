# Arbitrary Value Imputation and Code Explanation

This README provides a detailed explanation of Arbitrary Value Imputation as a missing data handling technique and describes each line of the provided Python code, including parameters and their roles.

## What is Arbitrary Value Imputation?
Arbitrary Value Imputation is a method to handle missing data by replacing missing values with specific arbitrary values. These values are chosen to distinguish imputed data from non-missing data, allowing analysts to track imputation. Common arbitrary values include values like -1, 99, or 999.

### Pros:
- Simple and fast to implement.
- Maintains dataset structure.

### Cons:
- Can distort data distribution.
- May introduce variance changes in the dataset.

## Code Explanation

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
```
- **pandas**: For data manipulation and analysis.
- **numpy**: Provides numerical operations.
- **matplotlib.pyplot**: For data visualization.
- **sklearn.model_selection.train_test_split**: Splits dataset into training and testing sets.
- **sklearn.impute.SimpleImputer**: Handles missing data imputation.
- **sklearn.compose.ColumnTransformer**: Applies transformations to specific columns.

### Loading the Dataset
```python
df = pd.read_csv('titanic_toy.csv')
df.head()
df.isnull().mean()
```
- **pd.read_csv('titanic_toy.csv')**: Reads the Titanic dataset into a DataFrame.
- **df.head()**: Displays the first 5 rows of the dataset.
- **df.isnull().mean()**: Calculates the proportion of missing values per column.

### Splitting the Data
```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- **X = df.drop(columns=['Survived'])**: Excludes the target variable ('Survived').
- **y = df['Survived']**: Sets 'Survived' as the target variable.
- **train_test_split**: Splits data into 80% training and 20% testing, ensuring reproducibility with **random_state=2**.

### Arbitrary Value Imputation (Manual)
```python
X_train['Age_99'] = X_train['Age'].fillna(99)
X_train['Age_minus1'] = X_train['Age'].fillna(-1)
X_train['Fare_999'] = X_train['Fare'].fillna(999)
X_train['Fare_minus1'] = X_train['Fare'].fillna(-1)
```
- **fillna(99)**: Fills missing values in the 'Age' column with 99.
- **fillna(-1)**: Fills missing values in the 'Age' column with -1.
- **fillna(999)**: Fills missing values in the 'Fare' column with 999.
- **fillna(-1)**: Fills missing values in the 'Fare' column with -1.

### Variance Analysis
```python
print('Original Age variable variance: ', X_train['Age'].var())
print('Age Variance after 99 wala imputation: ', X_train['Age_99'].var())
print('Age Variance after -1 wala imputation: ', X_train['Age_minus1'].var())

print('Original Fare variable variance: ', X_train['Fare'].var())
print('Fare Variance after 999 wala imputation: ', X_train['Fare_999'].var())
print('Fare Variance after -1 wala imputation: ', X_train['Fare_minus1'].var())
```
- **var()**: Computes variance to show how imputation impacts variability.

### Kernel Density Estimation (KDE) Plots
```python
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['Age'].plot(kind='kde', ax=ax)
X_train['Age_99'].plot(kind='kde', ax=ax, color='red')
X_train['Age_minus1'].plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
```
- **plot(kind='kde')**: Creates KDE plots to compare original and imputed distributions.
- **color**: Specifies line colors (e.g., red, green).
- **legend**: Adds a legend to differentiate distributions.

### Covariance and Correlation
```python
X_train.cov()
X_train.corr()
```
- **cov()**: Computes covariance matrix.
- **corr()**: Computes correlation matrix.

### Arbitrary Value Imputation (Using Scikit-learn)
```python
imputer1 = SimpleImputer(strategy='constant', fill_value=99)
imputer2 = SimpleImputer(strategy='constant', fill_value=999)
```
- **SimpleImputer(strategy='constant', fill_value=99)**: Replaces missing values in 'Age' with 99.
- **SimpleImputer(strategy='constant', fill_value=999)**: Replaces missing values in 'Fare' with 999.

### Column Transformer
```python
trf = ColumnTransformer([
    ('imputer1', imputer1, ['Age']),
    ('imputer2', imputer2, ['Fare'])
], remainder='passthrough')
```
- **ColumnTransformer**: Applies transformations to specified columns.
- **remainder='passthrough'**: Keeps untransformed columns as is.

### Fitting and Transforming
```python
trf.fit(X_train)
trf.named_transformers_['imputer1'].statistics_
trf.named_transformers_['imputer2'].statistics_
X_train = trf.transform(X_train)
X_test = trf.transform(X_test)
```
- **fit(X_train)**: Learns statistics for imputation.
- **named_transformers_**: Accesses individual transformers.
- **transform()**: Applies learned transformations to data.

### Output Transformed Data
```python
X_train
```
Displays the transformed training dataset.

---
This README explains how Arbitrary Value Imputation can be implemented manually and using Scikit-learn, providing insights into its effects on data distribution and variability.

