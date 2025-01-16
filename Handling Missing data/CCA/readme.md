# README.md

## Complete Case Analysis in Missing Data Handling

Complete Case Analysis (CCA) is a statistical technique used to handle missing data by removing any observations (rows) that have one or more missing values in the dataset. While this method is straightforward and easy to implement, it can lead to potential biases if the missing data is not completely random. Despite its simplicity, CCA is often used for its ease of interpretation and implementation.

This README explains the Python code provided for implementing CCA and visualizing its effects on a dataset.

---

### Code Explanation

#### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
- **`numpy`**: Used for numerical operations.
- **`pandas`**: Provides data structures like DataFrames for data manipulation and analysis.
- **`matplotlib.pyplot`**: Used for creating visualizations.

#### Load Dataset
```python
df = pd.read_csv('data_science_job.csv')
df.head()
```
- **`pd.read_csv('data_science_job.csv')`**: Loads the dataset from a CSV file into a pandas DataFrame.
- **`df.head()`**: Displays the first five rows of the DataFrame to preview the data.

#### Check for Missing Data
```python
df.isnull().mean()*100
df.shape
```
- **`df.isnull().mean()*100`**: Calculates the percentage of missing values for each column.
- **`df.shape`**: Returns the shape of the DataFrame (rows, columns).

#### Identify Columns with Missing Data
```python
cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
cols
```
- **List comprehension**: Identifies columns with less than 5% but more than 0% missing values.
- **`df.columns`**: Lists all column names in the DataFrame.
- **`df[var].isnull().mean()`**: Calculates the mean (percentage) of missing values in each column.

#### Sample the Selected Columns
```python
df[cols].sample(5)
```
- **`df[cols]`**: Filters the DataFrame to include only the identified columns.
- **`.sample(5)`**: Randomly selects five rows from the filtered DataFrame.

#### Examine Specific Column Values
```python
df['education_level'].value_counts()
```
- **`df['education_level']`**: Selects the `education_level` column.
- **`.value_counts()`**: Counts the frequency of each unique value in the column.

#### Calculate Retained Data After Dropping Rows
```python
len(df[cols].dropna()) / len(df)
```
- **`df[cols].dropna()`**: Removes rows with missing values from the selected columns.
- **`len(df[cols].dropna())`**: Counts the number of remaining rows.
- **`/ len(df)`**: Divides the retained row count by the total row count to calculate the percentage of retained data.

#### Create a New DataFrame After CCA
```python
new_df = df[cols].dropna()
df.shape, new_df.shape
```
- **`new_df = df[cols].dropna()`**: Stores the DataFrame after dropping rows with missing values.
- **`df.shape, new_df.shape`**: Compares the shape of the original and new DataFrames.

#### Visualize the Effect of CCA
```python
new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()
```
- **`new_df.hist()`**: Plots histograms for all numeric columns in `new_df`.
- **`bins=50`**: Sets the number of bins for the histogram.
- **`density=True`**: Normalizes the histogram.
- **`figsize=(12, 12)`**: Specifies the figure size.
- **`plt.show()`**: Displays the plot.

#### Compare Distributions Before and After CCA
The following sections compare distributions of specific columns before and after CCA.

**Training Hours**:
```python
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)
```
- **`ax = fig.add_subplot(111)`**: Creates a subplot.
- **`df['training_hours'].hist()`**: Plots the histogram for the original column in red.
- **`new_df['training_hours'].hist()`**: Plots the histogram for the column after CCA in green with transparency (`alpha=0.8`).

The same approach is repeated for `city_development_index` and `experience` columns.

#### Compare Category Distributions Before and After CCA
```python
temp = pd.concat([
    df['enrolled_university'].value_counts() / len(df),
    new_df['enrolled_university'].value_counts() / len(new_df)
], axis=1)

temp.columns = ['original', 'cca']
```
- **`pd.concat([...], axis=1)`**: Combines the value counts for each category before and after CCA into a single DataFrame.
- **`temp.columns = ['original', 'cca']`**: Renames the columns to indicate the original and post-CCA distributions.

The same approach is repeated for the `education_level` column.

---

### Summary
This code performs Complete Case Analysis on the dataset and provides visual and statistical insights into its effects. While CCA is easy to implement, users should be cautious about potential biases and data loss. Visualization and statistical comparison help understand the impact of missing data handling on the dataset.

