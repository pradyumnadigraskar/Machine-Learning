# Outlier Handling Using the IQR Proximity Rule

## Introduction
Outlier handling is an essential step in data preprocessing that helps improve model performance and ensures reliable statistical analysis. The **Interquartile Range (IQR) Proximity Rule** is a widely used method to detect and handle outliers. It uses the interquartile range to set boundaries beyond which data points are considered outliers.

This document explains the concept of outlier handling using the IQR Proximity Rule and provides a detailed line-by-line explanation of the code.

---

## What is the IQR Proximity Rule?
The IQR Proximity Rule identifies outliers based on the range of the middle 50% of the data. It calculates the **IQR** (Interquartile Range) as:

\[ \text{IQR} = Q3 - Q1 \]

Where:
- **Q1 (25th percentile):** The value below which 25% of the data lies.
- **Q3 (75th percentile):** The value below which 75% of the data lies.

Outliers are defined as data points that fall outside the range:
\[ \text{Lower Bound} = Q1 - 1.5 \times \text{IQR} \]
\[ \text{Upper Bound} = Q3 + 1.5 \times \text{IQR} \]

---

## Code Explanation

### Importing Libraries and Loading Data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('placement.csv')
df.head()
```
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `matplotlib.pyplot`: For visualization.
- `seaborn`: For advanced visualization.
- `pd.read_csv('placement.csv')`: Reads the dataset into a DataFrame named `df`.
- `df.head()`: Displays the first five rows of the DataFrame.

### Visualizing the Data
```python
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()
```
- `plt.figure(figsize=(16,5))`: Sets the overall figure size.
- `plt.subplot(1,2,1)`: Creates the first subplot for `cgpa` distribution.
- `sns.distplot(df['cgpa'])`: Plots the distribution of `cgpa`.
- `plt.subplot(1,2,2)`: Creates the second subplot for `placement_exam_marks` distribution.
- `sns.distplot(df['placement_exam_marks'])`: Plots the distribution of `placement_exam_marks`.
- `plt.show()`: Displays the visualizations.

### Summary and Boxplot
```python
df['placement_exam_marks'].describe()
sns.boxplot(df['placement_exam_marks'])
```
- `df['placement_exam_marks'].describe()`: Provides descriptive statistics (mean, min, max, etc.) for `placement_exam_marks`.
- `sns.boxplot(df['placement_exam_marks'])`: Creates a boxplot to visualize the spread and potential outliers.

### Calculating IQR and Bounds
```python
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)

iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print("Upper limit",upper_limit)
print("Lower limit",lower_limit)
```
- `quantile(0.25)` and `quantile(0.75)`: Calculate Q1 and Q3, respectively.
- `iqr = percentile75 - percentile25`: Computes the interquartile range.
- `upper_limit` and `lower_limit`: Calculate the boundaries for outlier detection using the IQR Proximity Rule.
- `print(...)`: Displays the calculated limits.

### Detecting Outliers
```python
df[df['placement_exam_marks'] > upper_limit]
df[df['placement_exam_marks'] < lower_limit]
```
- These lines filter and display rows where `placement_exam_marks` are above the upper limit or below the lower limit (outliers).

### Trimming Outliers
```python
new_df = df[df['placement_exam_marks'] < upper_limit]
new_df.shape
```
- `df[df['placement_exam_marks'] < upper_limit]`: Removes outliers above the upper limit.
- `new_df.shape`: Displays the shape of the trimmed dataset.

### Visualizing Trimmed Data
```python
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()
```
- Plots distributions and boxplots for the original and trimmed datasets to compare the effect of trimming.

### Capping Outliers
```python
new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)
```
- `new_df_cap = df.copy()`: Creates a copy of the original dataset.
- `np.where(condition, true, false)`: Replaces outliers with the nearest limit (capping).
  - `placement_exam_marks > upper_limit`: Values above the upper limit are capped at the upper limit.
  - `placement_exam_marks < lower_limit`: Values below the lower limit are capped at the lower limit.
  - Other values remain unchanged.

### Visualizing Capped Data
```python
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

plt.show()
```
- Plots distributions and boxplots for the original and capped datasets to compare the effect of capping.

---

## Conclusion
The IQR Proximity Rule is a robust method to detect and handle outliers in numerical data. This code demonstrates two common techniques:
1. **Trimming:** Removes outliers entirely from the dataset.
2. **Capping:** Replaces outliers with boundary values to retain all data points.

Both approaches have their use cases depending on the analysis or model requirements. Visualization before and after handling outliers provides insight into the data distribution and the effect of preprocessing.

