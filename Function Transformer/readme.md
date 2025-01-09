# README: Function Transformers and Code Explanation

## Table of Contents
1. [Introduction to Function Transformers](#introduction-to-function-transformers)
2. [Explanation of Code](#explanation-of-code)
3. [Function Transformers in Detail](#function-transformers-in-detail)
4. [Parameters Explained](#parameters-explained)

---

## Introduction to Function Transformers

A **FunctionTransformer** in scikit-learn allows you to transform your dataset using custom or pre-defined functions. It enables seamless integration of transformations, such as applying logarithms, scaling, or more complex operations, into a pipeline. These transformations are useful for ensuring that the data conforms to the requirements of the machine learning model, improving accuracy and interpretability.

---

## Explanation of Code

The provided code performs multiple tasks:

1. **Importing Necessary Libraries**
   ```python
   import pandas as pd
   import numpy as np
   import scipy.stats as stats
   import matplotlib.pyplot as plt
   import seaborn as sns

   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   from sklearn.model_selection import cross_val_score
   from sklearn.linear_model import LogisticRegression
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.preprocessing import FunctionTransformer
   from sklearn.compose import ColumnTransformer
   ```
   - `pandas`: For data manipulation and analysis.
   - `numpy`: For numerical operations.
   - `scipy.stats`: For statistical operations and plotting.
   - `matplotlib.pyplot` and `seaborn`: For data visualization.
   - `sklearn` modules: For machine learning models, transformations, and evaluations.

2. **Reading and Preprocessing Data**
   ```python
   df = pd.read_csv('train.csv',usecols=['Age','Fare','Survived'])
   df['Age'].fillna(df['Age'].mean(),inplace=True)
   ```
   - Reads the dataset `train.csv`, selecting only `Age`, `Fare`, and `Survived` columns.
   - Fills missing values in the `Age` column with the column mean.

3. **Splitting Data**
   ```python
   X = df.iloc[:,1:3]
   y = df.iloc[:,0]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - `X` includes `Fare` and `Survived` as features.
   - `y` contains the `Age` column as the target.
   - Splits the data into training (80%) and testing (20%) sets.

4. **Plotting Age Distribution**
   ```python
   plt.figure(figsize=(14,4))
   plt.subplot(121)
   sns.distplot(X_train['Age'])
   plt.title('Age PDF')

   plt.subplot(122)
   stats.probplot(X_train['Age'], dist="norm", plot=plt)
   plt.title('Age QQ Plot')
   plt.show()
   ```
   - Visualizes the probability density function (PDF) and quantile-quantile (QQ) plot for `Age`.

5. **Training and Evaluating Models**
   ```python
   clf = LogisticRegression()
   clf2 = DecisionTreeClassifier()

   clf.fit(X_train, y_train)
   clf2.fit(X_train, y_train)

   y_pred = clf.predict(X_test)
   y_pred1 = clf2.predict(X_test)

   print("Accuracy LR", accuracy_score(y_test, y_pred))
   print("Accuracy DT", accuracy_score(y_test, y_pred1))
   ```
   - Initializes logistic regression and decision tree classifiers.
   - Trains both models and evaluates their accuracy on the test set.

6. **Applying Log Transformation**
   ```python
   trf = FunctionTransformer(func=np.log1p)
   X_train_transformed = trf.fit_transform(X_train)
   X_test_transformed = trf.transform(X_test)
   ```
   - Applies a log transformation using `np.log1p` (log(x+1)) to handle skewness in the data.

7. **ColumnTransformer Example**
   ```python
   trf2 = ColumnTransformer([('log', FunctionTransformer(np.log1p), ['Fare'])], remainder='passthrough')
   X_train_transformed2 = trf2.fit_transform(X_train)
   X_test_transformed2 = trf2.transform(X_test)
   ```
   - Applies the log transformation only to the `Fare` column, leaving the rest unchanged.

8. **Custom Function Transformation**
   ```python
   def apply_transform(transform):
       X = df.iloc[:,1:3]
       y = df.iloc[:,0]
       trf = ColumnTransformer([('log', FunctionTransformer(transform), ['Fare'])], remainder='passthrough')
       X_trans = trf.fit_transform(X)

       clf = LogisticRegression()
       print("Accuracy", np.mean(cross_val_score(clf, X_trans, y, scoring='accuracy', cv=10)))

       plt.figure(figsize=(14,4))
       plt.subplot(121)
       stats.probplot(X['Fare'], dist="norm", plot=plt)
       plt.title('Fare Before Transform')
       plt.subplot(122)
       stats.probplot(X_trans[:,0], dist="norm", plot=plt)
       plt.title('Fare After Transform')
       plt.show()

   apply_transform(np.sin)
   ```
   - Defines a function to apply custom transformations (e.g., `np.sin`) and evaluate the model's accuracy.

---

## Function Transformers in Detail

### 1. `FunctionTransformer(func=np.log1p)`
Applies a logarithmic transformation (`log(x + 1)`) to reduce skewness in the data. Often used for highly skewed distributions.

### 2. `ColumnTransformer`
Allows selective transformation of specific columns while keeping the others unchanged. The `remainder='passthrough'` parameter ensures unselected columns remain in the output.

---

## Parameters Explained

### **FunctionTransformer**
- **`func`**: The transformation function (e.g., `np.log1p`, `np.sin`).
- **`inverse_func`**: (Optional) Inverse transformation function.
- **`validate`**: Whether to validate input dimensions.

### **ColumnTransformer**
- **`transformers`**: List of tuples defining transformations. Each tuple consists of:
  - Transformer name.
  - Transformer object.
  - List of columns to apply the transformer to.
- **`remainder`**: Specifies what to do with unselected columns (`'drop'` or `'passthrough'`).

