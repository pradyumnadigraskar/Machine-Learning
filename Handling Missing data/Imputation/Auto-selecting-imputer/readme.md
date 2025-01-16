# Auto-Select Imputer with Logistic Regression Pipeline

## Overview
This project demonstrates the use of auto-select imputer strategies and hyperparameter tuning using a machine learning pipeline. The dataset is preprocessed with a combination of numerical and categorical transformations, followed by training a Logistic Regression classifier. The best combination of preprocessing and model parameters is selected using GridSearchCV.

### What is an Auto-Select Imputer?
Auto-select imputation is a technique where the imputation strategy for missing values is chosen automatically through hyperparameter optimization. For example, different strategies like mean, median, or constant can be tested for numerical and categorical data to determine the best fit for the model.

---

## Code Explanation

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
```
- **Libraries Imported:**
  - `numpy` and `pandas`: For data manipulation.
  - `train_test_split` and `GridSearchCV`: For splitting data and hyperparameter tuning.
  - `ColumnTransformer` and `Pipeline`: To create preprocessing pipelines.
  - `SimpleImputer`, `StandardScaler`, `OneHotEncoder`: For data preprocessing.
  - `LogisticRegression`: The classifier used for prediction.

```python
df = pd.read_csv('train.csv')
df.head()
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
df.head()
```
- **Dataset Loading and Cleaning:**
  - The Titanic dataset is read from a CSV file.
  - Irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) are dropped.

```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
X_train.head()
```
- **Feature and Target Splitting:**
  - `Survived` is the target variable (`y`), and the rest are features (`X`).
  - Data is split into training and test sets (80% training, 20% testing).

```python
numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```
- **Numerical Feature Transformation:**
  - `Age` and `Fare` are defined as numerical features.
  - A pipeline is created for numerical transformation:
    - `SimpleImputer`: Fills missing values with the median.
    - `StandardScaler`: Scales the features to have zero mean and unit variance.

```python
categorical_features = ['Embarked', 'Sex']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',OneHotEncoder(handle_unknown='ignore'))
])
```
- **Categorical Feature Transformation:**
  - `Embarked` and `Sex` are defined as categorical features.
  - A pipeline is created for categorical transformation:
    - `SimpleImputer`: Fills missing values with the most frequent category.
    - `OneHotEncoder`: Converts categorical variables into one-hot encoded features.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```
- **ColumnTransformer:**
  - Combines numerical and categorical transformations into a single preprocessor.

```python
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```
- **Pipeline Creation:**
  - The pipeline includes:
    - `preprocessor`: Applies transformations to data.
    - `LogisticRegression`: Trains a logistic regression model.

```python
from sklearn import set_config
set_config(display='diagram')
clf
```
- **Pipeline Visualization:**
  - The pipeline structure is displayed as a diagram for better understanding.

```python
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
    'classifier__C': [0.1, 1.0, 10, 100]
}
```
- **Hyperparameter Grid Definition:**
  - Numerical imputer strategies: `mean`, `median`.
  - Categorical imputer strategies: `most_frequent`, `constant`.
  - Regularization strength (`C`) for logistic regression: `[0.1, 1.0, 10, 100]`.

```python
grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)
```
- **Grid Search with Cross-Validation:**
  - `GridSearchCV` tests all combinations of hyperparameters using 10-fold cross-validation.
  - The model is trained using the training data.

```python
print(f"Best params:")
print(grid_search.best_params_)
print(f"Internal CV score: {grid_search.best_score_:.3f}")
```
- **Best Parameters and Score:**
  - Displays the best combination of parameters and the corresponding cross-validation score.

```python
import pandas as pd
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[['param_classifier__C','param_preprocessor__cat__imputer__strategy','param_preprocessor__num__imputer__strategy','mean_test_score']]
```
- **Cross-Validation Results:**
  - Stores the results of cross-validation in a DataFrame.
  - Sorts the results by the mean test score in descending order.
  - Displays the top combinations of hyperparameters and their scores.

---

## Conclusion
This project demonstrates how to optimize preprocessing and model parameters for a logistic regression classifier using pipelines and GridSearchCV. The auto-select imputer strategy ensures optimal handling of missing data for both numerical and categorical features.

