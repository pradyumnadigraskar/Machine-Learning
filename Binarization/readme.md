# README.md

## What is Binarization?

Binarization is the process of converting continuous or multi-valued data into binary format (0 or 1). This is done based on a threshold value: values above the threshold are set to 1, and values below are set to 0. Binarization can be helpful in scenarios where binary representation simplifies computations or enhances interpretability in machine learning models.

### Why Do We Need Binarization?

1. **Feature Simplification**: Reduces complexity by representing data in a binary format.
2. **Improved Model Interpretability**: Easier to interpret binary features in decision trees, rule-based systems, or other models.
3. **Compatibility**: Some machine learning algorithms work better with binary inputs.
4. **Threshold-based Decisions**: Binarization makes it easier to define rules or decisions based on specific conditions.

### Libraries Used

1. **NumPy**: Provides numerical operations and array handling.
2. **Pandas**: Used for data manipulation and analysis.
3. **Scikit-learn**:
   - `Binarizer`: Transforms numerical data into binary format.
   - `DecisionTreeClassifier`: A classifier that splits data based on feature thresholds.
   - `ColumnTransformer`: Applies transformations to specific columns of the dataset.
   - `train_test_split`: Splits the dataset into training and testing subsets.
   - `cross_val_score`: Evaluates a model using cross-validation.
   - `accuracy_score`: Calculates the accuracy of predictions.

---

## Code Explanation

### Importing Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer
```
- **`numpy`** and **`pandas`**: Used for handling numerical data and data manipulation.
- **`sklearn` modules**: Provide tools for machine learning, preprocessing, and evaluation.

### Reading and Preprocessing Data
```python
df = pd.read_csv('train.csv')[['Age','Fare','SibSp','Parch','Survived']]
df.dropna(inplace=True)
df['family'] = df['SibSp'] + df['Parch']
df.drop(columns=['SibSp','Parch'], inplace=True)
```
1. **`pd.read_csv`**: Reads the dataset.
2. **`dropna`**: Removes rows with missing values to ensure clean data.
3. **`df['family'] = df['SibSp'] + df['Parch']`**: Creates a new column `family` as the sum of `SibSp` (siblings/spouses) and `Parch` (parents/children).
4. **`drop(columns=['SibSp','Parch'], inplace=True)`**: Removes `SibSp` and `Parch` columns since their information is combined into `family`.

### Splitting Data
```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
1. **`X`**: Features (independent variables).
2. **`y`**: Target variable (`Survived`).
3. **`train_test_split`**: Splits data into training (80%) and testing (20%) subsets.

### Training Without Binarization
```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy'))
```
1. **`DecisionTreeClassifier`**: Initializes a decision tree model.
2. **`fit`**: Trains the model on `X_train` and `y_train`.
3. **`predict`**: Predicts outcomes for `X_test`.
4. **`accuracy_score`**: Computes the accuracy of the predictions.
5. **`cross_val_score`**: Performs 10-fold cross-validation on the model to evaluate its performance.

### Applying Binarization
```python
trf = ColumnTransformer([
    ('bin', Binarizer(copy=False), ['family'])
], remainder='passthrough')
```
1. **`ColumnTransformer`**: Applies the `Binarizer` to the `family` column while leaving other columns unchanged (`remainder='passthrough'`).
2. **`Binarizer(copy=False)`**: Converts values in `family` to binary (0 or 1) without creating a copy for memory efficiency.

### Transforming Data
```python
X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)
pd.DataFrame(X_train_trf, columns=['family', 'Age', 'Fare'])
```
1. **`fit_transform`**: Fits the transformer on training data and applies the transformation.
2. **`transform`**: Applies the transformation to testing data.
3. **`pd.DataFrame`**: Creates a DataFrame to view the transformed data with appropriate column names.

### Training and Evaluating with Binarization
```python
clf = DecisionTreeClassifier()
clf.fit(X_train_trf, y_train)
y_pred2 = clf.predict(X_test_trf)
accuracy_score(y_test, y_pred2)
```
1. Similar steps as training without binarization but using transformed data (`X_train_trf` and `X_test_trf`).

### Cross-Validation with Binarization
```python
X_trf = trf.fit_transform(X)
np.mean(cross_val_score(DecisionTreeClassifier(), X_trf, y, cv=10, scoring='accuracy'))
```
1. **`fit_transform`**: Transforms the entire dataset.
2. **`cross_val_score`**: Evaluates the model with binarized features using 10-fold cross-validation.

---

## Summary
Binarization is a preprocessing technique that simplifies data representation and can improve machine learning performance. This code demonstrates its use with the `Binarizer` class in Scikit-learn, transforming the `family` column to binary format and comparing the model performance before and after binarization.

