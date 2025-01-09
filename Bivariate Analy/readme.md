# Bivariate Analysis with Python and Seaborn

This script demonstrates bivariate analysis using Python libraries such as Pandas and Seaborn. It explores datasets like `tips`, `titanic`, `flights`, and `iris` while visualizing relationships between different variables.

## What is Bivariate Analysis?
Bivariate analysis involves the analysis of two variables to determine the empirical relationship between them. It is a key statistical method to explore the correlation, trends, and patterns between variables. Techniques include scatter plots, bar plots, box plots, and more.

---

## Script Breakdown

### 1. Importing Libraries and Datasets
```python
import pandas as pd
import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()
```
- **Pandas**: Used for data manipulation and analysis.
- **Seaborn**: A Python data visualization library based on Matplotlib.
- **tips**: A dataset containing information about tips received in a restaurant.

### 2. Loading the Titanic Dataset
```python
titanic = pd.read_csv('train.csv')
```
- Loads the Titanic dataset for analysis.

### 3. Loading Other Datasets
```python
flights = sns.load_dataset('flights')
flights.head()
iris = sns.load_dataset('iris')
iris
```
- **flights**: A dataset about the number of passengers over time.
- **iris**: A dataset containing measurements of iris flower species.

### 4. Scatter Plot for Tips Dataset
```python
sns.scatterplot(tips['total_bill'], tips['tip'], hue=df['sex'], style=df['smoker'], size=df['size'])
```
- **Scatter Plot**: Visualizes the relationship between `total_bill` and `tip`.
- **Hue**: Differentiates data by the `sex` column.
- **Style**: Differentiates data by the `smoker` column.
- **Size**: Varies point size based on the `size` column.

### 5. Bar Plot for Titanic Dataset
```python
sns.barplot(titanic['Pclass'], titanic['Age'], hue=titanic['Sex'])
```
- **Bar Plot**: Shows the relationship between passenger class (`Pclass`) and age (`Age`), grouped by gender (`Sex`).

### 6. Box Plot for Titanic Dataset
```python
sns.boxplot(titanic['Sex'], titanic['Age'], hue=titanic['Survived'])
```
- **Box Plot**: Displays the distribution of age grouped by gender (`Sex`) and survival status (`Survived`).

### 7. Distribution Plot for Age and Survival
```python
sns.distplot(titanic[titanic['Survived']==0]['Age'], hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'], hist=False)
```
- **Distplot**: Compares the age distribution for passengers who survived (`Survived == 1`) and those who did not (`Survived == 0`).

### 8. Heatmap for Survival by Class
```python
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
```
- **Heatmap**: Shows the survival rate across different passenger classes.

### 9. Survival Rates by Embarkation Point
```python
(titanic.groupby('Embarked').mean()['Survived'] * 100)
```
- Computes the survival rate (in percentage) grouped by the embarkation point (`Embarked`).

### 10. Cluster Map for Parch and Survival
```python
sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))
```
- **Cluster Map**: Visualizes the relationship between the number of parents/children aboard (`Parch`) and survival status.

### 11. Pair Plot for Iris Dataset
```python
sns.pairplot(iris, hue='species')
```
- **Pair Plot**: Plots pairwise relationships between variables in the `iris` dataset, colored by species.

### 12. Line Plot for Flights Dataset
```python
new = flights.groupby('year').sum().reset_index()
sns.lineplot(new['year'], new['passengers'])
```
- **Line Plot**: Shows the trend in the number of passengers over the years.

### 13. Cluster Map for Flights Data
```python
sns.clustermap(flights.pivot_table(values='passengers', index='month', columns='year'))
```
- **Cluster Map**: Displays the number of passengers across months and years, grouped by similarities.

---

## Output Visualizations
- Scatter Plot: Relationship between `total_bill` and `tip`.
- Bar Plot: Distribution of age by passenger class and gender.
- Box Plot: Age distribution by survival status.
- Distplot: Comparison of age distribution for survival.
- Heatmap: Survival rate across passenger classes.
- Cluster Map: Relationship between variables using clustering techniques.
- Pair Plot: Pairwise relationships in the `iris` dataset.
- Line Plot: Passenger trends over the years.

---

## Notes
1. Ensure datasets (`train.csv`) are available in the working directory.
2. Replace `df` with the correct variable names (`tips` or `titanic`) to avoid errors in scatter plot and other visualizations.
3. Install the required libraries using:
   ```bash
   pip install pandas seaborn matplotlib
   ```

