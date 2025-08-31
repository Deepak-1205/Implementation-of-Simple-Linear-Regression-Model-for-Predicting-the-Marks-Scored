# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Deepak S
RegisterNumber: 212224230053  
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:


# DATASET
<img width="428" height="564" alt="image" src="https://github.com/user-attachments/assets/815e79bc-e026-475c-b0e1-7ec851c5eaf8" />



# HEAD VALUES
<img width="253" height="129" alt="416961128-622b828b-eba6-4aef-a6aa-22e274a4f704" src="https://github.com/user-attachments/assets/48b97ca7-5765-4a1b-8728-6b1ff73f43ac" />


# TAIL VALUES
<img width="255" height="131" alt="416961175-72767627-4499-4524-88e3-090b1051a565" src="https://github.com/user-attachments/assets/cbe5fae7-6a44-49fe-8e9c-25b00f438bf3" />


# X and Y values
<img width="714" height="589" alt="416961263-b49303b5-0a6f-4897-b8b9-4583ac7eb5e7" src="https://github.com/user-attachments/assets/0c22aad6-68c3-4163-9f9a-386ffad889d0" />


# Predication values of X and Y

<img width="712" height="80" alt="416961333-e6a1246a-7836-4dd9-8de1-8fe3595efce1" src="https://github.com/user-attachments/assets/8e4ed604-a1e8-4a35-93c5-3eab5fbf5570" />



# TRAINING SET
<img width="784" height="561" alt="416961391-9407cc8d-ca0e-4e20-ba2b-b605a5f703cc" src="https://github.com/user-attachments/assets/37f68d3b-0780-47f6-963d-87a5713414b4" />



# TESTING SET AND MSE,MAE and RMSE
<img width="913" height="660" alt="416961453-b4335ae9-f9b0-4884-ab62-2fbe37e4e54a" src="https://github.com/user-attachments/assets/abfbc854-8b27-4654-a49b-48250a7d22f9" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
