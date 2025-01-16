# Iterative Imputer Implementation

## Overview
Iterative Imputer is a sophisticated technique for handling missing data. Unlike simple imputation methods (e.g., mean, median, or mode), it models each feature with missing values as a function of other features and uses regression to estimate the missing values iteratively.

The algorithm works as follows:
1. Initialize missing values using simple imputation (e.g., mean or median).
2. For each feature with missing values:
    - Treat the feature as the target variable.
    - Use the other features as predictors.
    - Predict and update the missing values using the regression model.
3. Repeat this process for a predefined number of iterations or until convergence.

This README will explain the provided code, highlighting how iterative imputation is implemented manually using linear regression and multiple iterations.

---

## Code Explanation

### Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
```
- **pandas**: Used for data manipulation and analysis.
- **numpy**: Provides numerical operations and random number generation.
- **LinearRegression**: A machine learning algorithm for linear regression, used to predict missing values.

### Loading and Preprocessing Data
```python
df = np.round(pd.read_csv('50_Startups.csv')[['R&D Spend','Administration','Marketing Spend','Profit']]/10000)
np.random.seed(9)
df = df.sample(5)
df = df.iloc[:,0:-1]
```
- **`pd.read_csv('50_Startups.csv')`**: Reads the dataset.
- **Scaling**: The dataset is scaled down by dividing all values by 10,000 using `np.round()`.
- **Random Sampling**: Five random rows are selected for demonstration using `df.sample(5)`.
- **Removing the Target Column**: Drops the `Profit` column, focusing on predictors for this example.

### Introducing Missing Values
```python
df.iloc[1,0] = np.NaN
df.iloc[3,1] = np.NaN
df.iloc[-1,-1] = np.NaN
```
- Missing values are manually introduced into the dataset.

### Step 1: Initial Imputation with Mean
```python
df0 = pd.DataFrame()
df0['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].mean())
df0['Administration'] = df['Administration'].fillna(df['Administration'].mean())
df0['Marketing Spend'] = df['Marketing Spend'].fillna(df['Marketing Spend'].mean())
```
- Each column with missing values is imputed with the mean of that column using `.fillna(df[column].mean())`.

### Iterative Imputation Process
#### Iteration 1
```python
df1 = df0.copy()

# Predicting Missing Value in `R&D Spend`
df1.iloc[1,0] = np.NaN
X = df1.iloc[[0,2,3,4],1:3]  # Predictors (columns without missing values)
y = df1.iloc[[0,2,3,4],0]    # Target variable
lr = LinearRegression()
lr.fit(X, y)
df1.iloc[1,0] = lr.predict(df1.iloc[1,1:].values.reshape(1,2))
```
- Missing value in `R&D Spend` is predicted using a linear regression model.
- **`X`**: Predictor columns (`Administration` and `Marketing Spend`).
- **`y`**: Target column (`R&D Spend`).

```python
# Predicting Missing Value in `Administration`
df1.iloc[3,1] = np.NaN
X = df1.iloc[[0,1,2,4],[0,2]]  # Predictors
Y = df1.iloc[[0,1,2,4],1]      # Target variable
lr.fit(X, Y)
df1.iloc[3,1] = lr.predict(df1.iloc[3,[0,2]].values.reshape(1,2))
```
- The process is repeated for `Administration`, treating it as the target variable.

```python
# Predicting Missing Value in `Marketing Spend`
df1.iloc[4,-1] = np.NaN
X = df1.iloc[0:4,0:2]
y = df1.iloc[0:4,-1]
lr.fit(X, y)
df1.iloc[4,-1] = lr.predict(df1.iloc[4,0:2].values.reshape(1,2))
```
- Finally, `Marketing Spend` is imputed using the same process.

#### Iteration 2 and Beyond
```python
df2 = df1.copy()
# Repeat the process for all missing values
```
- The entire process is repeated for subsequent iterations.
- Each iteration refines the predictions for missing values.

### Convergence Check
```python
df2 - df1
df3 = df2.copy()
```
- The difference between iterations is computed (`df2 - df1`).
- If the differences are small or negligible, the imputation process is considered converged.

---

## Results
After multiple iterations, the imputed dataset is updated with accurate estimates for all missing values. This iterative process ensures that imputed values are consistent with the relationships among features.

---

## Advantages of Iterative Imputation
1. **Leverages Relationships**: Utilizes inter-feature dependencies for accurate imputation.
2. **Dynamic Adjustments**: Updates predictions iteratively, refining results.
3. **Customizable Models**: Allows for different regression models.

## Disadvantages
1. **Computationally Expensive**: Requires multiple iterations.
2. **Prone to Overfitting**: Can overfit to small datasets.

---

This manual implementation demonstrates the core concept of iterative imputation. In practice, libraries like `sklearn` offer built-in implementations such as `IterativeImputer` for ease and efficiency.
