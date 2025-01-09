# Univariate Analysis

## What is Univariate Analysis?
Univariate analysis involves examining a single variable to understand its distribution, central tendency, dispersion, and other statistical properties. This method is used to summarize and analyze patterns in a dataset to make inferences or identify anomalies.

### Key Components of Univariate Analysis:
- **Distribution**: How the values of a variable are spread (e.g., normal, skewed).
- **Central Tendency**: Measures such as mean, median, and mode.
- **Dispersion**: Measures like range, variance, and standard deviation.
- **Visualization**: Graphical representation through histograms, pie charts, boxplots, etc.

---

## Code Explanation

```python
import pandas as pd
import seaborn as sns
```
### Explanation:
- **`import pandas as pd`**: Imports the Pandas library for data manipulation and analysis.
- **`import seaborn as sns`**: Imports the Seaborn library for creating informative and attractive visualizations.

```python
df = pd.read_csv('train.csv')
```
### Explanation:
- Reads the dataset `train.csv` into a DataFrame named `df` using Pandas' `read_csv` function.

```python
sns.countplot(df['Embarked'])
```
### Explanation:
- Creates a count plot to show the frequency of each unique value in the `Embarked` column using Seaborn.
- **Purpose**: Visualizes the number of passengers who embarked from different locations.

```python
df['Sex'].value_counts().plot(kind='pie', autopct='%.2f')
```
### Explanation:
- Counts the unique values in the `Sex` column using `value_counts()`.
- Creates a pie chart using Matplotlib's `plot(kind='pie')`.
- **`autopct='%.2f'`**: Displays percentages with two decimal places.
- **Purpose**: Visualizes the proportion of males and females in the dataset.

```python
import matplotlib.pyplot as plt
plt.hist(df['Age'], bins=5)
```
### Explanation:
- **`import matplotlib.pyplot as plt`**: Imports Matplotlib for creating visualizations.
- **`plt.hist(df['Age'], bins=5)`**: Creates a histogram of the `Age` column, dividing the data into 5 bins.
- **Purpose**: Shows the distribution of ages in the dataset.

```python
sns.distplot(df['Age'])
```
### Explanation:
- Uses Seaborn's `distplot` to create a distribution plot for the `Age` column.
- **Purpose**: Visualizes the probability density of the `Age` column.

```python
sns.boxplot(df['Age'])
```
### Explanation:
- Creates a boxplot for the `Age` column using Seaborn's `boxplot` function.
- **Purpose**: Highlights the range, quartiles, and potential outliers in the `Age` column.

```python
df['Age'].min()
```
### Explanation:
- Computes the minimum value in the `Age` column.
- **Purpose**: Identifies the youngest passenger in the dataset.

```python
df['Age'].max()
```
### Explanation:
- Computes the maximum value in the `Age` column.
- **Purpose**: Identifies the oldest passenger in the dataset.

```python
df['Age'].mean()
```
### Explanation:
- Computes the mean (average) of the `Age` column.
- **Purpose**: Provides an estimate of the central tendency of the age distribution.

```python
df['Age'].skew()
```
### Explanation:
- Calculates the skewness of the `Age` column.
- **Purpose**: Measures the asymmetry of the age distribution. A skewness close to 0 indicates a symmetric distribution, positive values indicate a right skew, and negative values indicate a left skew.

---

## Summary of Techniques
1. **Count Plot**: Used to visualize categorical data (e.g., `Embarked`).
2. **Pie Chart**: Shows proportions (e.g., `Sex`).
3. **Histogram**: Visualizes frequency distribution (e.g., `Age`).
4. **Distribution Plot**: Highlights density and probability (e.g., `Age`).
5. **Boxplot**: Summarizes range, quartiles, and outliers (e.g., `Age`).
6. **Descriptive Statistics**: Provides summary metrics like min, max, mean, and skewness.

---

## Importance of Univariate Analysis
- **Data Cleaning**: Identifies missing or anomalous values.
- **Pattern Recognition**: Highlights trends and patterns.
- **Feature Engineering**: Helps in deriving new features or transforming existing ones.
- **Model Preparation**: Assists in understanding variable distributions for feature scaling and encoding.

By conducting univariate analysis, we gain valuable insights into individual variables, setting the stage for deeper multivariate analysis and predictive modeling.
