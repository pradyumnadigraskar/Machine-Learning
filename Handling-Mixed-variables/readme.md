# Handling Mixed Variables in Datasets

This guide explains how to handle mixed variables in datasets, including extracting and processing numerical and categorical parts. The example uses a Titanic dataset for demonstration.

## Prerequisites
- Python
- Pandas
- Matplotlib
- Numpy

## Code Explanation

```python
import numpy as np
import pandas as pd
```
- Importing necessary libraries:
  - `numpy`: For numerical operations.
  - `pandas`: For data manipulation and analysis.

```python
df = pd.read_csv('titanic.csv')
```
- Loading the Titanic dataset into a DataFrame named `df`.

### Handling Mixed Variables

#### Unique Values in `number` Column
```python
df['number'].unique()
```
- Finds unique values in the `number` column of the dataset.

#### Bar Plot of `number` Values
```python
fig = df['number'].value_counts().plot.bar()
fig.set_title('Passengers travelling with')
```
- Plots the frequency of unique values in the `number` column as a bar chart.
  - `value_counts()`: Counts occurrences of unique values.
  - `plot.bar()`: Creates a bar plot.
  - `set_title()`: Sets the title of the plot.

#### Extract Numerical and Categorical Parts from `number`
```python
df['number_numerical'] = pd.to_numeric(df['number'], errors='coerce', downcast='integer')
```
- Converts the `number` column to a numerical format.
  - `errors='coerce'`: Converts invalid parsing to `NaN`.
  - `downcast='integer'`: Converts data to the smallest integer subtype to save memory.

```python
df['number_categorical'] = np.where(df['number_numerical'].isnull(), df['number'], np.nan)
```
- Creates a new column `number_categorical` that stores non-numeric values from the `number` column.
  - `np.where(condition, x, y)`: Returns `x` if `condition` is true, otherwise `y`.

#### Handling Cabin Data
```python
df['Cabin'].unique()
```
- Lists unique values in the `Cabin` column.

```python
df['cabin_num'] = df['Cabin'].str.extract('(\d+)')
```
- Extracts numerical parts from the `Cabin` column.
  - `str.extract('(\d+)')`: Uses a regular expression to extract digits.

```python
df['cabin_cat'] = df['Cabin'].str[0]
```
- Captures the first letter of each entry in the `Cabin` column.

#### Bar Plot of `cabin_cat`
```python
df['cabin_cat'].value_counts().plot(kind='bar')
```
- Visualizes the distribution of `cabin_cat` as a bar plot.
  - `kind='bar'`: Specifies the type of plot as a bar chart.

#### Handling Ticket Data
```python
df['Ticket'].unique()
```
- Lists unique values in the `Ticket` column.

```python
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
```
- Extracts the last part of the `Ticket` column as a number.
  - `apply(lambda s: s.split()[-1])`: Splits the string and selects the last part.

```python
df['ticket_num'] = pd.to_numeric(df['ticket_num'], errors='coerce', downcast='integer')
```
- Converts the extracted ticket number to numeric format.

```python
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
```
- Extracts the first part of the `Ticket` column as a category.

```python
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan, df['ticket_cat'])
```
- Replaces numeric values in `ticket_cat` with `NaN` to keep only categorical data.

### Final Outputs
```python
df['ticket_cat'].unique()
```
- Displays unique values in the `ticket_cat` column.

## Summary
This process demonstrates how to:
1. Separate numerical and categorical data in mixed columns.
2. Use regex and string operations for extraction.
3. Visualize data distributions using bar plots.
4. Handle missing or invalid data using `NaN` values.

By processing mixed variables effectively, you can prepare datasets for further analysis or machine learning tasks.

