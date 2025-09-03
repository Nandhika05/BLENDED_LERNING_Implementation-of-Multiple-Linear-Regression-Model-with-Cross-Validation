# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data

Load the dataset using Pandas.
Drop unnecessary columns like car_ID and CarName.
Convert categorical variables into numerical format using one-hot encoding.

2. Split the Data

Separate the dataset into features (X) and target variable (y).
Split the dataset into training and testing sets using train_test_split.

3.Build and Train the Model

Create a LinearRegression model instance.
Fit the model on the training data.

4. Evaluate the Model

Perform 5-fold cross-validation using cross_val_score.
Evaluate the model on the test set using Mean Squared Error (MSE) and R² score.
Visualize the actual vs predicted car prices using a scatter plot.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Nandhika P
RegisterNumber: 212223040125
*/
```
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1.load and prepare data
data = pd.read_csv("CarPrice_Assignment.csv")

# Simple preprocessing
data = data.drop(['car_ID','CarName'],axis=1)
data = pd.get_dummies(data,drop_first=True)

#2. Split data
x = data.drop('price',axis=1)
y = data['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#3. create and train model
model = LinearRegression()
model.fit(x_train,y_train)

#4. Evaluate with cross-validation
print('Name: Nandhika P')
print('Reg. No: 212223040125')
print("\n===Cross Validation ===")
cv_scores = cross_val_score(model, x, y, cv=5)
print("Fold R² Scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R² Score: {cv_scores.mean():.4f}")

#5. test set evaluation
y_pred = model.predict(x_test)
print("\n===Test Set Perfomance ===")
print(f"MSE:{mean_squared_error(y_test,y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")

#6. visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actua1 vs Predicted Prices")
plt .grid(True)
plt. show( )
```

## Output:

<img width="551" height="139" alt="image" src="https://github.com/user-attachments/assets/3350d5ef-8408-4c8e-850c-06d635133496" />

<img width="309" height="110" alt="image" src="https://github.com/user-attachments/assets/cf4593b4-2980-4b9d-9cd8-0e55cb58e231" />

<img width="542" height="430" alt="image" src="https://github.com/user-attachments/assets/944b493e-8018-4fa6-bd56-7f3d60d4dcad" />

## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
