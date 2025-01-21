# README: Feature Construction and Splitting with Logistic Regression

## Overview
This README explains the process of feature construction and splitting using a Titanic dataset (`train.csv`). The steps include data preprocessing, feature engineering, and model evaluation using Logistic Regression. Each line of code and parameter is detailed for better understanding.

---

## Prerequisites
Ensure the following libraries are installed:
- `numpy`
- `pandas`
- `seaborn`
- `scikit-learn`

---

## Code Explanation

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns
```
- `numpy`: Provides numerical operations and array handling.
- `pandas`: Handles data manipulation and analysis.
- `sklearn.model_selection.cross_val_score`: Used for evaluating model performance with cross-validation.
- `sklearn.linear_model.LogisticRegression`: Implements logistic regression for classification tasks.
- `seaborn`: Visualization library (imported but not used in this script).

### 2. Loading and Preprocessing Data
```python
df = pd.read_csv('train.csv')[['Age','Pclass','SibSp','Parch','Survived']]
df.head()
df.dropna(inplace=True)
df.head()
```
- `pd.read_csv('train.csv')`: Loads the dataset. Selects only the relevant columns (`Age`, `Pclass`, `SibSp`, `Parch`, and `Survived`).
- `df.dropna(inplace=True)`: Removes rows with missing values to ensure clean data.

### 3. Splitting Features (X) and Target (y)
```python
X = df.iloc[:,0:4]
y = df.iloc[:,-1]
X.head()
```
- `df.iloc[:,0:4]`: Extracts the first four columns as features (X).
- `df.iloc[:,-1]`: Extracts the last column (`Survived`) as the target variable (y).

### 4. Model Evaluation
```python
np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20))
```
- `cross_val_score`: Performs cross-validation to evaluate model accuracy.
  - `LogisticRegression()`: Model used for classification.
  - `scoring='accuracy'`: Metric for evaluation.
  - `cv=20`: Uses 20-fold cross-validation.
- `np.mean()`: Calculates the mean accuracy across folds.

### 5. Feature Construction
```python
X['Family_size'] = X['SibSp'] + X['Parch'] + 1
X.head()
```
- `X['Family_size']`: Constructs a new feature by summing `SibSp` (siblings/spouses) and `Parch` (parents/children) and adding 1 (the passenger).

#### Adding Family Type
```python
def myfunc(num):
    if num == 1:
        return 0  # Alone
    elif num > 1 and num <= 4:
        return 1  # Small family
    else:
        return 2  # Large family

X['Family_type'] = X['Family_size'].apply(myfunc)
X.head()
```
- `myfunc`: A helper function to classify family size into categories:
  - `0`: Alone
  - `1`: Small family (2-4 members)
  - `2`: Large family (>4 members)
- `X['Family_size'].apply(myfunc)`: Applies `myfunc` to each value in the `Family_size` column.

#### Dropping Redundant Columns
```python
X.drop(columns=['SibSp','Parch','Family_size'], inplace=True)
X.head()
```
- `X.drop`: Removes unnecessary columns after constructing the `Family_type` feature.

#### Reevaluating the Model
```python
np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20))
```
- Re-runs cross-validation with the updated feature set.

### 6. Feature Splitting
```python
df = pd.read_csv('train.csv')
df.head()
```
- Reloads the original dataset.

#### Extracting Titles
```python
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df[['Title','Name']]
```
- `df['Name'].str.split(', ', expand=True)[1]`: Splits the `Name` column at `, ` and selects the second part.
- `.str.split('.', expand=True)[0]`: Further splits the result at `.` and extracts the title.

#### Calculating Survival Rates by Title
```python
(df.groupby('Title').mean()['Survived']).sort_values(ascending=False)
```
- `df.groupby('Title')`: Groups data by title.
- `.mean()['Survived']`: Calculates the mean survival rate for each title.
- `.sort_values(ascending=False)`: Sorts titles by survival rate in descending order.

#### Adding Marital Status
```python
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
```
- `df['Is_Married']`: Initializes a new column indicating marital status.
- `.loc[df['Title'] == 'Mrs']`: Sets `Is_Married` to 1 for rows where the title is `Mrs` (married women).

---

## Summary
This script demonstrates:
- **Feature Construction**: Adding meaningful derived features (`Family_size`, `Family_type`).
- **Feature Splitting**: Extracting titles and marital status from names.
- **Model Evaluation**: Using Logistic Regression with cross-validation to measure performance.

