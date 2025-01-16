# Outlier Handling Using Percentile Method

## Introduction
Outlier handling is a critical preprocessing step in data analysis and machine learning to ensure the robustness of the models. The **Percentile Method** is one technique used to detect and handle outliers by capping the values within a specific percentile range. This README explains how the Percentile Method is applied, providing a detailed explanation of the code and parameters.

---

## What is the Percentile Method?
The **Percentile Method** is a statistical approach where values in a dataset are capped or removed based on specified upper and lower percentiles. For instance, to remove extreme outliers, one may consider capping values below the 1st percentile and above the 99th percentile.

### Steps in the Percentile Method:
1. **Identify Percentile Thresholds**: Calculate the 1st percentile (lower limit) and the 99th percentile (upper limit).
2. **Filter or Cap Values**:
   - **Filtering**: Remove data points outside the threshold range.
   - **Capping (Winsorization)**: Replace outlier values with the respective threshold values.

---

## Code Explanation

### 1. Import Necessary Libraries
```python
import numpy as np
import pandas as pd
```
These libraries are essential for handling data (`pandas`) and performing numerical operations (`numpy`).

---

### 2. Load Dataset
```python
df = pd.read_csv('weight-height.csv')
df.head()
df.shape
```
- `pd.read_csv('weight-height.csv')`: Loads the dataset into a Pandas DataFrame.
- `df.head()`: Displays the first five rows of the dataset for inspection.
- `df.shape`: Returns the number of rows and columns in the dataset.

---

### 3. Statistical Summary of the `Height` Column
```python
df['Height'].describe()
```
- Provides a summary of the `Height` column, including count, mean, standard deviation, min, max, and percentile values (25th, 50th, and 75th).

---

### 4. Visualize the Distribution and Boxplot
```python
import seaborn as sns
sns.distplot(df['Height'])
sns.boxplot(df['Height'])
```
- `sns.distplot(df['Height'])`: Plots the distribution of the `Height` column to visualize its spread.
- `sns.boxplot(df['Height'])`: Generates a boxplot to identify potential outliers.

---

### 5. Calculate Percentile Thresholds
```python
upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)
upper_limit
lower_limit
```
- `df['Height'].quantile(0.99)`: Computes the 99th percentile of the `Height` column, defining the upper limit.
- `df['Height'].quantile(0.01)`: Computes the 1st percentile of the `Height` column, defining the lower limit.

---

### 6. Filter Outliers by Thresholds
```python
new_df = df[(df['Height'] <= upper_limit) & (df['Height'] >= lower_limit)]
new_df['Height'].describe()
df['Height'].describe()
```
- `df[(df['Height'] <= upper_limit) & (df['Height'] >= lower_limit)]`: Filters the dataset to retain rows where the `Height` values lie within the specified percentile range.
- `new_df['Height'].describe()`: Displays the statistical summary of the filtered dataset.
- `df['Height'].describe()`: Displays the original dataset's statistical summary for comparison.

---

### 7. Visualize Filtered Data
```python
sns.distplot(new_df['Height'])
sns.boxplot(new_df['Height'])
```
- `sns.distplot(new_df['Height'])`: Visualizes the distribution of the filtered dataset.
- `sns.boxplot(new_df['Height'])`: Plots a boxplot for the filtered dataset to confirm outlier removal.

---

### 8. Perform Capping (Winsorization)
```python
df['Height'] = np.where(
    df['Height'] >= upper_limit,
    upper_limit,
    np.where(
        df['Height'] <= lower_limit,
        lower_limit,
        df['Height']
    )
)
```
- `np.where(condition, true_value, false_value)`:
  - Replaces `Height` values above the upper limit with the upper limit value.
  - Replaces `Height` values below the lower limit with the lower limit value.
  - Retains values within the range as they are.

---

### 9. Summary and Visualization Post-Capping
```python
df.shape
df['Height'].describe()
sns.distplot(df['Height'])
sns.boxplot(df['Height'])
```
- `df.shape`: Confirms the dataset's dimensions remain unchanged post-capping.
- `df['Height'].describe()`: Displays the statistical summary after Winsorization.
- `sns.distplot(df['Height'])`: Visualizes the modified distribution of the `Height` column.
- `sns.boxplot(df['Height'])`: Confirms outlier capping via a boxplot.

---

## Key Insights:
- **Percentile Method** ensures extreme outliers are handled without altering the overall data distribution significantly.
- **Filtering** removes outliers but reduces the dataset size.
- **Capping (Winsorization)** retains all data points by replacing extreme values with threshold limits.

---

## Conclusion
The Percentile Method is a simple yet effective way to handle outliers. It is particularly useful when dealing with continuous data where extreme values can skew analysis or affect model performance.
