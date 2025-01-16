# Mean-Median Imputer: Explanation and Code Walkthrough

## What is Mean-Median Imputation?
Mean-median imputation is a simple technique to handle missing data in a dataset. It involves replacing missing values in a column with either the **mean** or the **median** of that column. This method is often used for numerical data to maintain consistency and avoid issues caused by missing values.

### Key Characteristics:
- **Mean Imputation**: Replaces missing values with the arithmetic average of the available data in the column.
- **Median Imputation**: Replaces missing values with the middle value when the data is sorted.
- **Advantages**: Easy to implement, computationally inexpensive.
- **Disadvantages**: Can distort data distribution and variance, especially if the data contains outliers.

---

## Code Walkthrough

### Importing Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
```
- `pandas`: Used for data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib.pyplot`: For plotting and visualizations.
- `train_test_split`: Splits the dataset into training and testing subsets.
- `SimpleImputer`: Implements mean/median imputation.
- `ColumnTransformer`: Applies different transformations to different columns in the dataset.

### Loading the Dataset
```python
df = pd.read_csv('titanic_toy.csv')
df.head()
```
- Loads the Titanic dataset and displays the first few rows for inspection.

### Exploring the Data
```python
df.info()
df.isnull().mean()
```
- `df.info()`: Provides a summary of the dataset, including data types and missing values.
- `df.isnull().mean()`: Calculates the proportion of missing values for each column.

### Splitting Data into Features and Target
```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```
- `X`: Feature variables (all columns except `Survived`).
- `y`: Target variable (`Survived`).
- `train_test_split`: Splits the data into 80% training and 20% testing sets.

### Handling Missing Data Manually
#### Calculating Mean and Median
```python
mean_age = X_train['Age'].mean()
median_age = X_train['Age'].median()

mean_fare = X_train['Fare'].mean()
median_fare = X_train['Fare'].median()
```
- `mean_age` and `median_age`: Compute the mean and median of the `Age` column.
- `mean_fare` and `median_fare`: Compute the mean and median of the `Fare` column.

#### Imputing Missing Values
```python
X_train['Age_median'] = X_train['Age'].fillna(median_age)
X_train['Age_mean'] = X_train['Age'].fillna(mean_age)

X_train['Fare_median'] = X_train['Fare'].fillna(median_fare)
X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare)
```
- `fillna()`: Replaces missing values in `Age` and `Fare` columns with the median and mean values calculated earlier.

#### Comparing Variance
```python
print('Original Age variable variance: ', X_train['Age'].var())
print('Age Variance after median imputation: ', X_train['Age_median'].var())
print('Age Variance after mean imputation: ', X_train['Age_mean'].var())

print('Original Fare variable variance: ', X_train['Fare'].var())
print('Fare Variance after median imputation: ', X_train['Fare_median'].var())
print('Fare Variance after mean imputation: ', X_train['Fare_mean'].var())
```
- Compares variance before and after imputation for both `Age` and `Fare` columns to observe the impact of imputation.

#### Visualizing Distributions
```python
fig = plt.figure()
ax = fig.add_subplot(111)

X_train['Age'].plot(kind='kde', ax=ax)
X_train['Age_median'].plot(kind='kde', ax=ax, color='red')
X_train['Age_mean'].plot(kind='kde', ax=ax, color='green')
ax.legend(loc='best')

fig = plt.figure()
ax = fig.add_subplot(111)

X_train['Fare'].plot(kind='kde', ax=ax)
X_train['Fare_median'].plot(kind='kde', ax=ax, color='red')
X_train['Fare_mean'].plot(kind='kde', ax=ax, color='green')
ax.legend(loc='best')
```
- Plots Kernel Density Estimation (KDE) graphs to visualize the distribution of `Age` and `Fare` before and after imputation.

#### Covariance and Correlation
```python
X_train.cov()
X_train.corr()
```
- `cov()`: Computes the covariance matrix of the dataset.
- `corr()`: Computes the correlation matrix of the dataset.

#### Boxplot Comparisons
```python
X_train[['Age', 'Age_median', 'Age_mean']].boxplot()
X_train[['Fare', 'Fare_median', 'Fare_mean']].boxplot()
```
- Displays boxplots to compare the original and imputed values.

### Using Sklearn for Imputation
#### Setting Up Imputers
```python
imputer1 = SimpleImputer(strategy='median')
imputer2 = SimpleImputer(strategy='mean')
```
- `SimpleImputer`: Configures imputers with specified strategies (`median` and `mean`).

#### Applying ColumnTransformer
```python
trf = ColumnTransformer([
    ('imputer1', imputer1, ['Age']),
    ('imputer2', imputer2, ['Fare'])
], remainder='passthrough')
```
- `ColumnTransformer`: Applies different transformations to specified columns:
  - `imputer1`: Median imputation for the `Age` column.
  - `imputer2`: Mean imputation for the `Fare` column.

#### Fitting and Transforming Data
```python
trf.fit(X_train)
trf.named_transformers_['imputer1'].statistics_
trf.named_transformers_['imputer2'].statistics_
X_train = trf.transform(X_train)
X_test = trf.transform(X_test)
```
- `fit`: Calculates the statistics (mean/median) required for imputation.
- `transform`: Replaces missing values in `X_train` and `X_test` with the calculated statistics.

---

## Conclusion
Mean-median imputation is a straightforward and effective technique for handling missing values in numerical data. While easy to implement, it can affect data variance and should be used carefully, especially when the dataset contains significant outliers.
