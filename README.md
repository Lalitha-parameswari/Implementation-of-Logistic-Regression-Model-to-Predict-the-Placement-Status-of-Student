# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LALITHA PARAMESWARI.C
RegisterNumber:212219220027

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
*/
```

## Output:

## Original data(first five columns): 

![image](https://user-images.githubusercontent.com/103946827/174729801-16dda6d1-f8ea-49e3-96b7-ff8370e4fb6e.png)

## Data after dropping unwanted columns(first five):

![image](https://user-images.githubusercontent.com/103946827/174729940-76b876c2-4138-45d0-95b6-ec88d757b744.png)

## Checking the presence of null values:

![image](https://user-images.githubusercontent.com/103946827/174730170-f0825b19-76af-47f3-a1c8-c59f1f34e3e8.png)

## Checking the presence of duplicated values:

![image](https://user-images.githubusercontent.com/103946827/174730343-46f8a078-d067-4be4-801a-12e69ab3377e.png)

## Data after Encoding:

![image](https://user-images.githubusercontent.com/103946827/174730425-7e0b6acf-fa17-44f4-98ad-3e3a400e0a1e.png)

## X Data:

![image](https://user-images.githubusercontent.com/103946827/174730525-8d883447-a59a-405e-85ba-158ef682fdb9.png)

## Y Data:

![image](https://user-images.githubusercontent.com/103946827/174730698-1a29e6ab-2b9a-40b5-b0e5-b965191a3e6c.png)

## Predicted Values:

![image](https://user-images.githubusercontent.com/103946827/174730777-c9e91daa-3a4d-4e02-8415-accc6f26818c.png)

## Accuracy Score:

![image](https://user-images.githubusercontent.com/103946827/174730844-a0e1fc4b-cdb7-4e1d-a07b-3674019c63b9.png)

## Confusion Matrix:

![image](https://user-images.githubusercontent.com/103946827/174730916-d0afa09f-e12f-4be0-b80e-c2e2a7f6eeb8.png)

## Classification Report:

![image](https://user-images.githubusercontent.com/103946827/174730988-7fb5a695-89df-4c80-8fa2-bd2fd72a7184.png)

## Predicting output from Regression Model:

![image](https://user-images.githubusercontent.com/103946827/174731147-6ed1dcc8-9714-4433-b3a8-3f426ca41e7c.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
