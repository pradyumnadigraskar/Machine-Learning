# README: Simple Linear Regression and Custom Implementation

## Overview
This README explains **Simple Linear Regression** in detail and walks through a custom implementation of linear regression using Python. The code is designed to demonstrate how to fit a linear model to data and make predictions. Each line of code and parameter is explained for clarity.

---

## What is Simple Linear Regression?

Simple Linear Regression is a statistical method used to model the relationship between two variables:
- **Independent Variable (X)**: The input or predictor variable.
- **Dependent Variable (y)**: The output or response variable.

The relationship is modeled by fitting a straight line:

\[ y = mX + b \]

Where:
- \( m \): Slope of the line (rate of change of y with respect to X).
- \( b \): Intercept (value of y when X = 0).

The goal is to minimize the difference (error) between the predicted and actual values of \( y \).

---

## Code Explanation

### 1. Custom Class for Linear Regression
```python
class MeraLR:
    
    def __init__(self):
        self.m = None
        self.b = None
```
- **`class MeraLR:`**: A custom class to implement linear regression.
- **`__init__`:** Initializes the model parameters:
  - `self.m`: Slope of the line (initialized as `None`).
  - `self.b`: Intercept of the line (initialized as `None`).

### 2. Fitting the Model
```python
    def fit(self,X_train,y_train):
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):
            
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
        
        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)       
```
- **`fit`**: Calculates the slope and intercept using the training data:
  - **`X_train`**: Independent variable (training data).
  - **`y_train`**: Dependent variable (training data).
- **`num`**: Numerator of the slope formula:
  \[ \text{num} = \sum{(X_i - \bar{X})(y_i - \bar{y})} \]
- **`den`**: Denominator of the slope formula:
  \[ \text{den} = \sum{(X_i - \bar{X})^2} \]
- **`self.m`**: Slope of the line:
  \[ m = \frac{\text{num}}{\text{den}} \]
- **`self.b`**: Intercept of the line:
  \[ b = \bar{y} - m\bar{X} \]
- **`print(self.m)`** and **`print(self.b)`**: Outputs the calculated slope and intercept.

### 3. Making Predictions
```python
    def predict(self,X_test):
        
        print(X_test)
        
        return self.m * X_test + self.b
```
- **`predict`**: Predicts the value of y for given X values.
  - **`X_test`**: Independent variable (test data).
- Returns the predicted y values:
  \[ \hat{y} = mX + b \]

---

### 4. Loading the Data
```python
import numpy as np
import pandas as pd
df = pd.read_csv('placement.csv')
df.head()
```
- **`import numpy`** and **`import pandas`**: Imports required libraries for numerical operations and data manipulation.
- **`pd.read_csv('placement.csv')`**: Loads the dataset from a CSV file named `placement.csv`.
- **`df.head()`**: Displays the first five rows of the dataset.

### 5. Extracting Features and Target
```python
X = df.iloc[:,0].values
y = df.iloc[:,1].values
```
- **`X`**: Extracts the first column as the independent variable (e.g., scores).
- **`y`**: Extracts the second column as the dependent variable (e.g., placements).

### 6. Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
```
- **`train_test_split`**: Splits the data into training and testing sets:
  - `test_size=0.2`: 20% of the data is used for testing.
  - `random_state=2`: Ensures reproducibility by fixing the random seed.

### 7. Initializing and Training the Model
```python
lr = MeraLR()
lr.fit(X_train,y_train)
```
- **`lr = MeraLR()`**: Creates an instance of the `MeraLR` class.
- **`lr.fit(X_train, y_train)`**: Trains the model using the training data.

### 8. Verifying the Training Data
```python
X_train.shape[0]
X_train[0]
X_train.mean()
X_test[0]
```
- **`X_train.shape[0]`**: Returns the number of training samples.
- **`X_train[0]`**: Retrieves the first element of the training data.
- **`X_train.mean()`**: Calculates the mean of the training data.
- **`X_test[0]`**: Retrieves the first element of the test data.

### 9. Making Predictions
```python
print(lr.predict(X_test[0]))
```
- **`lr.predict(X_test[0])`**: Predicts the dependent variable for the first test sample.
- **`print`**: Outputs the predicted value.

---

## Summary
This script demonstrates:
- **Custom Linear Regression**: Building a regression model from scratch.
- **Feature Extraction and Splitting**: Preparing data for training and testing.
- **Model Evaluation**: Fitting the model and predicting outcomes.

You can adapt this code for any dataset with a linear relationship between two variables. Replace `placement.csv` with your dataset, ensuring it has two columns for X and y.

