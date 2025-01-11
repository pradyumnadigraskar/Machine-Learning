# Discretization (Binning) and Code Explanation

## **Discretization or Binning**

Discretization (or binning) is the process of converting continuous features into discrete bins or intervals. It is widely used in data preprocessing to simplify analysis and improve interpretability, especially when working with machine learning algorithms that perform better with categorical data.

### **Types of Binning**
There are three primary strategies for binning:

1. **Uniform Binning**
   - Divides the data into equal-width intervals.
   - The range of the feature is split into equally spaced bins, regardless of the data distribution.

   **Technique**:
   - Specify the number of bins (`n_bins`).
   - The width of each bin is calculated as:
     \[ \text{Bin Width} = \frac{\text{Max Value} - \text{Min Value}}{n_{bins}} \]

2. **Quantile Binning**
   - Divides the data into bins with approximately equal numbers of data points.
   - Useful when you want to ensure each bin contains a similar number of samples.

   **Technique**:
   - Bins are created based on percentiles of the data.
   - Specify the number of bins (`n_bins`).

3. **K-Means Binning**
   - Uses the K-Means clustering algorithm to group data into bins based on similarity.
   - Dynamically identifies clusters in the data rather than relying on fixed widths or percentiles.

   **Technique**:
   - Specify the number of clusters (`n_bins`).
   - The centroids of the K-Means algorithm define the bin edges.

---

## **Code Explanation**

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
```
- **`pandas`**: Used for data manipulation and analysis.
- **`numpy`**: Provides support for numerical computations.
- **`matplotlib.pyplot`**: Visualization library for plotting histograms.
- **`train_test_split`**: Splits the dataset into training and testing subsets.
- **`DecisionTreeClassifier`**: Machine learning model for classification.
- **`accuracy_score`**: Measures the accuracy of the model.
- **`cross_val_score`**: Evaluates model performance using cross-validation.
- **`KBinsDiscretizer`**: Discretizes continuous features into discrete bins.
- **`ColumnTransformer`**: Applies transformations to specific columns.

### Reading and Preprocessing the Dataset
```python
df = pd.read_csv('train.csv', usecols=['Age', 'Fare', 'Survived'])
df.dropna(inplace=True)
```
- Reads the CSV file with columns `Age`, `Fare`, and `Survived`.
- Drops rows with missing values to ensure clean data for processing.

### Splitting Features and Target
```python
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **`X`**: Independent variables (`Fare` and `Survived`).
- **`y`**: Target variable (`Age`).
- Splits the dataset into training (80%) and testing (20%) sets using `train_test_split`.

### Training a Decision Tree Classifier
```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
```
- Initializes the decision tree classifier.
- Trains the model on the training data.
- Predicts outcomes for the test set.
- Computes the accuracy of predictions.

### Cross-Validation
```python
np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy'))
```
- Evaluates the model using 10-fold cross-validation.
- **`cv=10`**: Splits the dataset into 10 subsets.
- **`scoring='accuracy'`**: Metric for evaluation.

### Discretization with `KBinsDiscretizer`
```python
kbin_age = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
kbin_fare = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile')
trf = ColumnTransformer([
    ('first', kbin_age, [0]),
    ('second', kbin_fare, [1])
])
```
- **`KBinsDiscretizer`**: Discretizes features into bins.
  - **`n_bins=15`**: Specifies 15 bins.
  - **`encode='ordinal'`**: Outputs discrete integers.
  - **`strategy='quantile'`**: Uses quantile-based binning.
- **`ColumnTransformer`**: Applies separate transformations to `Age` and `Fare`.

### Transforming Data
```python
X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)
```
- Transforms `X_train` and `X_test` using the specified binning strategy.

### Viewing Bin Edges
```python
trf.named_transformers_['first'].bin_edges_
```
- Retrieves the bin edges for the `Age` feature.

### Outputting Binned Data
```python
output = pd.DataFrame({
    'age': X_train['Age'],
    'age_trf': X_train_trf[:, 0],
    'fare': X_train['Fare'],
    'fare_trf': X_train_trf[:, 1]
})
output['age_labels'] = pd.cut(x=X_train['Age'], bins=trf.named_transformers_['first'].bin_edges_[0].tolist())
output['fare_labels'] = pd.cut(x=X_train['Fare'], bins=trf.named_transformers_['second'].bin_edges_[0].tolist())
```
- **`pd.cut`**: Creates labeled intervals for the binned features.

### Updated Decision Tree Classifier
```python
clf = DecisionTreeClassifier()
clf.fit(X_train_trf, y_train)
y_pred2 = clf.predict(X_test_trf)
accuracy_score(y_test, y_pred2)
```
- Trains the classifier on discretized data.

### Visualizing Discretization
```python
def discretize(bins, strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    trf = ColumnTransformer([
        ('first', kbin_age, [0]),
        ('second', kbin_fare, [1])
    ])
    X_trf = trf.fit_transform(X)
    print(np.mean(cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy')))
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(X['Age'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:, 0], color='red')
    plt.title("After")
    plt.show()
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.hist(X['Fare'])
    plt.title("Before")
    plt.subplot(122)
    plt.hist(X_trf[:, 1], color='red')
    plt.title("Fare")
    plt.show()

discretize(5, 'kmeans')
```
- Visualizes the effect of discretization on `Age` and `Fare` features using histograms.
- **`bins`**: Number of bins for discretization.
- **`strategy`**: Strategy used for binning (`uniform`, `quantile`, `kmeans`).

