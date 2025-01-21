# Principal Component Analysis (PCA)

## Overview
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms data into a lower-dimensional space while retaining as much variance as possible. PCA identifies the directions (principal components) in which the data varies the most and projects the data onto these directions, reducing the number of features while preserving essential information.

## Code Explanation

### Libraries Imported
```python
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
```
- **numpy**: Used for mathematical operations and array handling.
- **pandas**: For data manipulation and analysis.
- **plotly.express**: For creating interactive visualizations.
- **sklearn.preprocessing.StandardScaler**: For standardizing the dataset.
- **matplotlib**: For visualizing data in 2D and 3D.

### Step 1: Generating Data
```python
np.random.seed(23)

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

df = pd.DataFrame(class1_sample,columns=['feature1','feature2','feature3'])
df['target'] = 1

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

df1 = pd.DataFrame(class2_sample,columns=['feature1','feature2','feature3'])
df1['target'] = 0

df = df.append(df1,ignore_index=True)

df = df.sample(40)
df.head()
```
- `np.random.seed(23)`: Ensures reproducibility by setting a fixed seed.
- `np.random.multivariate_normal`: Generates random samples from a multivariate normal distribution.
  - **Parameters:**
    - `mu_vec1` / `mu_vec2`: Mean vectors of the distributions.
    - `cov_mat1` / `cov_mat2`: Covariance matrices.
    - `20`: Number of samples to generate for each class.
- `pd.DataFrame`: Creates a DataFrame for each class's samples with features `feature1`, `feature2`, and `feature3`.
- `df['target']`: Adds a target column with labels 1 for `class1_sample` and 0 for `class2_sample`.
- `df.append(df1, ignore_index=True)`: Combines both datasets.
- `df.sample(40)`: Shuffles the rows randomly.

### Step 2: Visualizing the Data
```python
fig = px.scatter_3d(df, x=df['feature1'], y=df['feature2'], z=df['feature3'],
              color=df['target'].astype('str'))
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig.show()
```
- `px.scatter_3d`: Creates a 3D scatter plot.
  - **Parameters:**
    - `x`, `y`, `z`: Axes mapped to features.
    - `color`: Target column to differentiate classes.
- `update_traces`: Modifies the marker's size and border.

### Step 3: Standard Scaling
```python
scaler = StandardScaler()
df.iloc[:,0:3] = scaler.fit_transform(df.iloc[:,0:3])
```
- `StandardScaler`: Standardizes features to have mean 0 and variance 1.
  - **Parameters:**
    - `fit_transform`: Computes mean and standard deviation, then applies scaling.
  - `df.iloc[:,0:3]`: Selects feature columns.

### Step 4: Covariance Matrix
```python
covariance_matrix = np.cov([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]])
print('Covariance Matrix:\n', covariance_matrix)
```
- `np.cov`: Computes the covariance matrix of the standardized features.
  - **Parameters:**
    - `df.iloc[:,0:3]`: Input features.

### Step 5: Eigenvalues and Eigenvectors
```python
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_values
eigen_vectors
```
- `np.linalg.eig`: Computes eigenvalues and eigenvectors.
  - **Eigenvalues:** Represent the variance along each principal component.
  - **Eigenvectors:** Directions of the principal components.

### Step 6: Visualizing Eigenvectors
```python
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
```
- Custom class `Arrow3D` creates 3D arrows for eigenvector visualization.

```python
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(df['feature1'], df['feature2'], df['feature3'], 'o', markersize=8, color='blue', alpha=0.2)
ax.plot([df['feature1'].mean()], [df['feature2'].mean()], [df['feature3'].mean()], 'o', markersize=10, color='red', alpha=0.5)
for v in eigen_vectors.T:
    a = Arrow3D([df['feature1'].mean(), v[0]], [df['feature2'].mean(), v[1]], [df['feature3'].mean(), v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show()
```
- `ax.plot`: Plots features and their mean points.
- `Arrow3D`: Plots eigenvectors as arrows.

### Step 7: Transforming Data
```python
pc = eigen_vectors[0:2]
transformed_df = np.dot(df.iloc[:,0:3],pc.T)
new_df = pd.DataFrame(transformed_df,columns=['PC1','PC2'])
new_df['target'] = df['target'].values
```
- `np.dot`: Projects the data onto the first two principal components.
- `pd.DataFrame`: Stores transformed data with `PC1` and `PC2` as columns.

### Step 8: Visualizing Transformed Data
```python
fig = px.scatter(x=new_df['PC1'],
                 y=new_df['PC2'],
                 color=new_df['target'],
                 color_discrete_sequence=px.colors.qualitative.G10
                )

fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()
```
- `px.scatter`: Creates a 2D scatter plot with transformed features.
- `color_discrete_sequence`: Defines the color palette for classes.

## Summary
This code demonstrates PCA step-by-step, from generating a dataset to visualizing the reduced dimensions. PCA helps simplify complex datasets while retaining essential patterns.



# Principal Component Analysis (PCA) Implementation

## Overview
This project demonstrates the implementation of Principal Component Analysis (PCA) in Python. PCA is a powerful dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while retaining the most significant variance in the data. The goal of this project is to provide a step-by-step guide to PCA, from generating synthetic data to visualizing the results.

## Features
- **Data Generation:** Create synthetic 3D data samples from two multivariate normal distributions.
- **Data Preprocessing:** Standardize the dataset to ensure uniform scaling.
- **Covariance Matrix:** Compute the covariance matrix of the dataset.
- **Eigenvalues and Eigenvectors:** Calculate the eigenvalues and eigenvectors for dimensionality reduction.
- **Data Transformation:** Project the data onto the principal components.
- **Visualization:** Visualize both the original 3D data and the transformed 2D data using:
  - Interactive 3D scatter plots with Plotly.
  - 2D scatter plots of the principal components.

## Installation
To run this project, ensure you have Python installed along with the required libraries:

```bash
pip install numpy pandas plotly scikit-learn matplotlib
```

## Usage
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Run the Script:**
   Execute the Python script containing the PCA implementation.

3. **View Results:**
   - Visualizations of the data before and after PCA will be displayed.
   - Analyze the variance explained by each principal component.

## Files
- `pca_script.py`: Main script implementing PCA.
- `README.md`: Documentation of the project.

## Implementation Steps

### Step 1: Generate Data
Two synthetic datasets are created using multivariate normal distributions with specific mean vectors and covariance matrices.

### Step 2: Visualize 3D Data
Interactive 3D scatter plots allow you to explore the original data distribution.

### Step 3: Standardize Data
Standardization ensures all features contribute equally to the PCA process.

### Step 4: Compute Covariance Matrix
The covariance matrix captures the relationships between features.

### Step 5: Eigen Decomposition
Eigenvalues and eigenvectors are computed from the covariance matrix to identify principal components.

### Step 6: Project Data
Data is transformed to a lower-dimensional space using the top principal components.

### Step 7: Visualize Transformed Data
Scatter plots of the transformed 2D data illustrate the effect of PCA.

## Visualization
- **3D Plot:** Original data points in 3D space.
- **Eigenvectors:** Visualized as arrows indicating directions of maximum variance.
- **2D Plot:** Transformed data projected onto the first two principal components.

## Results
- PCA reduces dimensionality while preserving the most important features of the data.
- Visualizations highlight the separation of classes in the reduced space.

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, contact [Your Name] at [Your Email].

