# K-Nearest Neighbors (KNN) Imputer

## What is KNN Imputer?
KNN Imputer is a machine learning-based technique used for handling missing values in a dataset. It imputes missing values by identifying the `k` nearest neighbors of the missing data point based on other features in the dataset and uses their values to estimate the missing value. The similarity between data points is calculated using a distance metric, such as Euclidean distance.

The KNN Imputer is particularly effective when the missing values are correlated with other features in the dataset.

---

## Working of KNN Imputer
1. **Identify Missing Values:** The KNN Imputer detects rows with missing values in the dataset.
2. **Find Neighbors:** For each missing value, the algorithm finds the `k` nearest neighbors (rows without missing values) based on the selected features and distance metric.
3. **Weighted Average (or Mean):** The missing value is imputed using the weighted average or mean of the corresponding feature values from the `k` neighbors. The weights are determined by the inverse of the distance, i.e., closer neighbors have higher influence.

---

## Code Explanation
The code demonstrates the usage of the KNN Imputer and compares it with a Simple Imputer (using the mean strategy).

### Step-by-Step Explanation

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
- **Importing Libraries:**
  - `numpy` and `pandas` are used for data manipulation and analysis.
  - `train_test_split` from `sklearn` is used to split the dataset into training and testing sets.
  - `KNNImputer` and `SimpleImputer` handle missing values.
  - `LogisticRegression` is the classification model.
  - `accuracy_score` evaluates the model’s performance.

---

```python
df = pd.read_csv('train.csv')[['Age','Pclass','Fare','Survived']]
df.head()
```
- **Load Dataset:** Reads the dataset from `train.csv` and selects only the columns `Age`, `Pclass`, `Fare`, and `Survived`.
- **Purpose:** These columns are used to train a logistic regression model to predict survival.

---

```python
df.isnull().mean() * 100
```
- **Check Missing Values:** Calculates the percentage of missing values in each column.

---

```python
X = df.drop(columns=['Survived'])
y = df['Survived']
```
- **Define Features (`X`) and Target (`y`):**
  - `X` contains the independent variables: `Age`, `Pclass`, and `Fare`.
  - `y` contains the target variable: `Survived`.

---

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train.head()
```
- **Split Data:** Splits the data into training (80%) and testing (20%) sets using a random state for reproducibility.

---

```python
knn = KNNImputer(n_neighbors=3, weights='distance')
```
- **Initialize KNN Imputer:**
  - `n_neighbors=3`: The algorithm considers the 3 nearest neighbors for imputing missing values.
  - `weights='distance'`: Assigns weights inversely proportional to the distance. Closer neighbors contribute more to the imputed value.

---

```python
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)
```
- **Apply KNN Imputer:**
  - `fit_transform` imputes missing values in the training data.
  - `transform` applies the same imputation logic to the test data.

---

```python
lr = LogisticRegression()

lr.fit(X_train_trf, y_train)
```
- **Train Logistic Regression Model:**
  - Fits the logistic regression model using the imputed training data (`X_train_trf`) and corresponding labels (`y_train`).

---

```python
y_pred = lr.predict(X_test_trf)

accuracy_score(y_test, y_pred)
```
- **Make Predictions:**
  - Predicts survival using the imputed test data (`X_test_trf`).
  - Calculates accuracy by comparing predictions (`y_pred`) with actual labels (`y_test`).

---

### Comparison with Simple Imputer

```python
si = SimpleImputer()

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)
```
- **Simple Imputer:**
  - Imputes missing values using the mean of each column.
  - `fit_transform` applies imputation on training data.
  - `transform` applies the imputation logic to test data.

---

```python
lr = LogisticRegression()

lr.fit(X_train_trf2, y_train)

y_pred2 = lr.predict(X_test_trf2)

accuracy_score(y_test, y_pred2)
```
- **Train and Evaluate Logistic Regression:**
  - Uses the Simple Imputer-transformed data (`X_train_trf2` and `X_test_trf2`) to train and test the logistic regression model.
  - Calculates accuracy of the model.

---

## Results
- **KNN Imputer:** Provides more nuanced imputations by considering feature correlations and distances between rows, often leading to better model performance.
- **Simple Imputer:** Averages missing values but does not consider feature relationships.

The comparison highlights how the choice of imputation method can impact the performance of the machine learning model.

