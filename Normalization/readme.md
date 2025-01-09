# README: Data Normalization and Code Explanation

## What is Normalization?
Normalization is a data preprocessing technique used to scale data to a specific range, typically [0, 1]. It ensures that all features contribute equally to the analysis or model training, preventing features with larger scales from dominating the results. Normalization is particularly useful when using machine learning models that are sensitive to the magnitude of feature values.

---

## Code Explanation

### Libraries Imported
```python
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for statistical data visualization
```
- `numpy`: Provides tools for numerical computations.
- `pandas`: Used to handle and manipulate datasets.
- `matplotlib.pyplot` & `seaborn`: For creating data visualizations.

### Loading and Preparing the Dataset
```python
df = pd.read_csv('wine_data.csv',header=None,usecols=[0,1,2])
df.columns=['Class label', 'Alcohol', 'Malic acid']
```
- Reads the `wine_data.csv` file.
- Specifies columns 0, 1, and 2 to use.
- Renames columns for better understanding: 'Class label', 'Alcohol', and 'Malic acid'.

### Initial Visualization
```python
sns.kdeplot(df['Alcohol'])
sns.kdeplot(df['Malic acid'])
```
- Plots the kernel density estimate (KDE) for `Alcohol` and `Malic acid` to understand their distributions.

```python
color_dict={1:'red',3:'green',2:'blue'}
sns.scatterplot(df['Alcohol'],df['Malic acid'],hue=df['Class label'],palette=color_dict)
```
- Defines a color dictionary for class labels.
- Creates a scatterplot for `Alcohol` vs. `Malic acid` with colors representing class labels.

### Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class label', axis=1),
                                                    df['Class label'],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape
```
- Splits the data into training and test sets (70% train, 30% test).
- Ensures the random split is reproducible with `random_state=0`.

### Applying Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```
- Imports `MinMaxScaler` to perform normalization.

```python
scaler.fit(X_train)
```
- Fits the scaler to the training data, calculating the min and max values for each feature.

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- Transforms the training and test sets based on the scaling parameters.

```python
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```
- Converts the scaled data back to a DataFrame for easier handling and visualization.

### Comparing Before and After Normalization
```python
np.round(X_train.describe(), 1)
np.round(X_train_scaled.describe(), 1)
```
- Displays summary statistics of the original and normalized training data to compare their scales.

### Visualizing Scaling Effects
#### Scatterplot Before and After Scaling
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Alcohol'], X_train['Malic acid'],c=y_train)
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Alcohol'], X_train_scaled['Malic acid'],c=y_train)
ax2.set_title("After Scaling")
plt.show()
```
- Shows how the scale of `Alcohol` and `Malic acid` changes after normalization.

#### KDE Plots Before and After Scaling
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)
sns.kdeplot(X_train['Malic acid'], ax=ax1)

ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()
```
- Visualizes the distributions of `Alcohol` and `Malic acid` before and after scaling.

#### Individual KDE Plots for Features
```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.set_title('Alcohol Distribution Before Scaling')
sns.kdeplot(X_train['Alcohol'], ax=ax1)

ax2.set_title('Alcohol Distribution After Standard Scaling')
sns.kdeplot(X_train_scaled['Alcohol'], ax=ax2)
plt.show()
```
- Focuses on the distribution of `Alcohol` before and after scaling.

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.set_title('Malic acid Distribution Before Scaling')
sns.kdeplot(X_train['Malic acid'], ax=ax1)

ax2.set_title('Malic acid Distribution After Standard Scaling')
sns.kdeplot(X_train_scaled['Malic acid'], ax=ax2)
plt.show()
```
- Focuses on the distribution of `Malic acid` before and after scaling.

---

## Summary
This code demonstrates the process of normalization using `MinMaxScaler`. It highlights the impact of scaling on data distributions and how it helps to standardize the range of features, ensuring consistent input to machine learning models. Visualizations are used to effectively compare the distributions and scales before and after normalization.

