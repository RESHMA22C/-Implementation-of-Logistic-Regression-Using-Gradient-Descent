# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Load and Preprocess Data Load dataset from CSV.
Drop irrelevant columns: sl_no, salary.

Convert all categorical columns to numeric using label encoding.

2.Prepare Features and Target Separate the dataset into:
Features X (all columns except status)

Target y (status)

Scale features using StandardScaler to normalize input values.

3.Add Bias Term Add a column of 1s to X for the intercept (bias) term in the model.

4.Split Data Use train_test_split() to divide X and y into:

Training set (X_train, y_train)

Test set (X_test, y_test)

5.Initialize Model Set initial weights theta to zeros with shape (number_of_features + 1, 1).

6.Define Logistic Regression Components Sigmoid Function: 𝜎 ( 𝑧 ) = 1 1

𝑒 − 𝑧 σ(z)= 1+e −z

1
# 𝐽 ( 𝜃 )
# − 1 𝑚 ∑ [ 𝑦 ⋅ log ⁡ ( ℎ ) + ( 1 − 𝑦 ) ⋅ log ⁡ ( 1 − ℎ ) ] J(θ)=− m 1​∑[y⋅log(h)+(1−y)⋅log(1−h)] Where ℎ​
𝜎 ( 𝑋 ⋅ 𝜃 ) h=σ(X⋅θ)

7.Train with Gradient Descent Loop for a number of iterations (e.g., 1000)

# Compute predictions ℎ
𝜎 ( 𝑋 ⋅ 𝜃 ) h=σ(X⋅θ)

Calculate gradient:

# gradient
1 𝑚 ⋅ 𝑋 𝑇 ⋅ ( ℎ − 𝑦 ) gradient= m 1​⋅X T ⋅(h−y) Update weights:

# 𝜃 :
𝜃 − 𝛼 ⋅ gradient θ:=θ−α⋅gradient Optionally print the loss at every 100 iterations

8.Make Predictions Predict class labels by:
Computing probabilities using sigmoid

Assigning class 1 if probability ≥ 0.5, else 0

9.Evaluate Model Compute accuracy:
# Accuracy
Number of Correct Predictions Total Predictions Accuracy= Total Predictions Number of Correct Predictions​

10.Predict on New Data Input a new sample of student data.
Apply the same scaling and bias addition.

Use trained theta to predict placement status.

Output result as Placed or Not Placed.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: RESHMA C
RegisterNumber:  212223040168
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/82febc69-3d3c-44c2-909b-7b0847c9b19c)

![image](https://github.com/user-attachments/assets/f5882be9-a6df-4226-b064-2075831cd6e2)

![image](https://github.com/user-attachments/assets/030c80d4-1e80-462f-ac04-93a2a29198fe)

![image](https://github.com/user-attachments/assets/cbd3cfb8-8389-414b-80f8-7125f33d1189)

Accuracy and Predicted value
![image](https://github.com/user-attachments/assets/ce67ee7e-d2ca-472b-8265-1ace7f528dbd)

Predicted value
![image](https://github.com/user-attachments/assets/a89da52b-0315-4949-acea-04b70be4b6fc)

![image](https://github.com/user-attachments/assets/892eb005-809f-4061-aeec-dbdf26b414eb)

![image](https://github.com/user-attachments/assets/9f8a4e5a-42df-4589-9480-e4ade2a5761d)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

