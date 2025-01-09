# Handling Missing Data and Encoding Features in Machine Learning

This script demonstrates two approaches for handling missing data and encoding categorical variables: the **Normal Transformer** method and the **Column Transformer** method. Both methods prepare data for machine learning models, but they differ in execution and scalability.

---

## Differences Between Normal and Column Transformers

### Normal Transformer
- Each transformation is applied step-by-step to individual columns or groups of columns.
- Requires manual application of transformations for both training and test datasets.
- Provides fine-grained control over each transformation.

### Column Transformer
- Combines all transformations into a single pipeline.
- Automates the process of applying the same transformations to both training and test datasets.
- More scalable and compact, especially for datasets with multiple features.

---

## Script Breakdown

### Importing Libraries
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
```
- **Numpy and Pandas**: Used for data manipulation.
- **SimpleImputer**: Handles missing values by imputing them.
- **OneHotEncoder**: Encodes categorical variables as binary columns.
- **OrdinalEncoder**: Encodes ordinal variables with a meaningful order.

### Loading and Exploring the Dataset
```python
df = pd.read_csv('covid_toy.csv')
df.head()
df.isnull().sum()
```
- Reads the dataset `covid_toy.csv`.
- Displays the first few rows of the dataset.
- Checks for missing values in each column using `isnull().sum()`.

### Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['has_covid']),
    df['has_covid'],
    test_size=0.2
)
```
- Splits the data into training and test sets.
- `X_train`, `X_test`: Features for training and testing.
- `y_train`, `y_test`: Target variable (whether the patient has COVID).

### Normal Transformer
#### Imputing Missing Values in `fever`
```python
si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.fit_transform(X_test[['fever']])
```
- **SimpleImputer**: Replaces missing values in the `fever` column with a default strategy (mean by default).
- Fits the imputer on training data and applies it to both training and test data.

#### Encoding Ordinal Variable `cough`
```python
oe = OrdinalEncoder(categories=[['Mild', 'Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])
X_test_cough = oe.fit_transform(X_test[['cough']])
```
- **OrdinalEncoder**: Converts `cough` into numeric values based on the specified order: 'Mild' < 'Strong'.
- Fits the encoder on training data and applies it to both training and test data.

#### Encoding One-Hot Variables `gender` and `city`
```python
ohe = OneHotEncoder(drop='first', sparse=False)
X_train_gender_city = ohe.fit_transform(X_train[['gender', 'city']])
X_test_gender_city = ohe.fit_transform(X_test[['gender', 'city']])
```
- **OneHotEncoder**: Converts `gender` and `city` into binary columns.
- Drops the first category (`drop='first'`) to avoid multicollinearity.

### Column Transformer
```python
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ('tnf1', SimpleImputer(), ['fever']),
        ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
        ('tnf3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city'])
    ],
    remainder='passthrough'
)
```
- Combines all transformations into a single pipeline:
  - `tnf1`: Imputes missing values in `fever`.
  - `tnf2`: Encodes `cough` as ordinal.
  - `tnf3`: Applies one-hot encoding to `gender` and `city`.
- `remainder='passthrough'`: Keeps columns that are not specified in transformations.

#### Applying Column Transformer
```python
transformer.fit_transform(X_train).shape
transformer.transform(X_test).shape
```
- **fit_transform**: Fits the transformer to `X_train` and transforms it.
- **transform**: Applies the same transformations to `X_test`.

---

## Output and Insights
- **Normal Transformer**: Provides manual control but requires repetitive code for training and test data.
- **Column Transformer**: Simplifies the process and is more efficient for large datasets.

---

## Notes
1. Ensure `covid_toy.csv` is available in the working directory.
2. Install required libraries using:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Use Column Transformer for scalable and maintainable code, especially for large or complex datasets.

