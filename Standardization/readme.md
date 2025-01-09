# README.md

## Standardization and Feature Engineering

### What is Standardization?
Standardization is the process of scaling features in a dataset to have a mean of 0 and a standard deviation of 1. This ensures that features contribute equally to the model and prevents bias caused by features with larger magnitudes dominating the learning process.

Formula for standardization:
\[
x_{standardized} = \frac{x - \mu}{\sigma}
\]
Where:
- \( x \) is the original value
- \( \mu \) is the mean of the feature
- \( \sigma \) is the standard deviation of the feature

### What is Feature Engineering?
Feature engineering involves transforming raw data into meaningful input features that improve the performance of machine learning models. This process includes selecting, modifying, and creating new features.

### Importance of Scaling
- **Improves model convergence**: Models like logistic regression or neural networks perform better when data is standardized.
- **Reduces bias**: Features with large magnitudes don't dominate others.
- **Better performance with distance-based algorithms**: Algorithms like SVM and kNN are sensitive to feature magnitudes.

### Feature Engineering vs. Scaling
| Feature Engineering             | Scaling                 |
|---------------------------------|-------------------------|
| Focuses on creating new features or modifying existing ones to capture relevant patterns. | Focuses on normalizing or standardizing feature values. |
| Example: Creating a feature `Age_group` from `Age`. | Example: Standardizing `Age` to have mean 0 and std. dev. 1. |
| A broad term covering transformations, encoding, and scaling. | A specific transformation under feature engineering. |

---

## Code Explanation

### Importing Required Libraries
```python
import numpy as np  # Linear algebra operations
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Statistical data visualization
```

### Loading and Preprocessing Data
```python
df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:, 2:]
```
- Load the dataset from a CSV file.
- Select relevant columns starting from the 3rd column onwards.

### Splitting Data into Training and Testing Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1), df['Purchased'], test_size=0.3, random_state=0
)
```
- Split the dataset into training (70%) and testing (30%) sets.
- `X_train` and `X_test`: Input features.
- `y_train` and `y_test`: Target variable.

### Standardizing Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```
- **`StandardScaler`**: Scales features to have mean 0 and standard deviation 1.
- **Fit**: Learns scaling parameters from the training set.
- **Transform**: Applies the learned parameters to scale both training and testing sets.

### Visualizing Scaling Effects
#### Scatter Plot
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'], color='red')
ax2.set_title("After Scaling")
plt.show()
```
- **Before Scaling**: Shows original feature magnitudes.
- **After Scaling**: Displays standardized feature magnitudes.

#### Kernel Density Estimation (KDE) Plots
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_scaled['Age'], ax=ax2)
sns.kdeplot(X_train_scaled['EstimatedSalary'], ax=ax2)
plt.show()
```
- Visualizes the distribution of features before and after scaling.

### Why Scaling is Important?
#### Logistic Regression Example
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(X_train, y_train)
lr_scaled.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)

from sklearn.metrics import accuracy_score
print("Actual", accuracy_score(y_test, y_pred))
print("Scaled", accuracy_score(y_test, y_pred_scaled))
```
- Model performance improves when features are standardized, especially for linear models like logistic regression.

#### Decision Tree Classifier Example
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt_scaled = DecisionTreeClassifier()

dt.fit(X_train, y_train)
dt_scaled.fit(X_train_scaled, y_train)

y_pred = dt.predict(X_test)
y_pred_scaled = dt_scaled.predict(X_test_scaled)

print("Actual", accuracy_score(y_test, y_pred))
print("Scaled", accuracy_score(y_test, y_pred_scaled))
```
- Decision trees are less affected by scaling, as they split data based on feature thresholds.

### Effect of Outliers
```python
df = df.append(pd.DataFrame({'Age': [5, 90, 95], 'EstimatedSalary': [1000, 250000, 350000], 'Purchased': [0, 1, 1]}), ignore_index=True)
plt.scatter(df['Age'], df['EstimatedSalary'])
```
- Adding outliers skews the distribution and affects scaling. Visualize outliers with a scatter plot.

### Additional Splitting and Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1), df['Purchased'], test_size=0.3, random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Re-splits the updated dataset and applies standardization.

---

## Key Takeaways
1. **Standardization** is essential for improving model convergence and ensuring fair contributions from features.
2. **Feature Engineering** enhances model performance by capturing relevant patterns.
3. **Scaling and outlier handling** significantly impact model performance.
4. Models like logistic regression rely on scaled data for accurate results, while tree-based models are less affected.

