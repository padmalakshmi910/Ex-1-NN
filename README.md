<H3>G.Padma Lakshmi</H3>
<H3>212225230206.</H3>
<H3>EX. NO.1</H3>
<H3>29-07-2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

~~~

## STEP 1: Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Libraries Imported Successfully")


## STEP 2: Importing the dataset

dataset = pd.read_csv("Churn_Modelling.csv")

print("\nDataset Loaded Successfully")
print(dataset.head())

print("\nDataset Shape")
print(dataset.shape)

print("\nDataset Information")
print(dataset.info())

print("\nStatistical Summary")
print(dataset.describe())

# Selecting Features and Target Variable

# Removing unnecessary columns
X = dataset.iloc[:, 3:13].values

# Target variable
y = dataset.iloc[:, 13].values

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)


## STEP 3: Taking care of missing data

print("\nChecking Missing Values")

print(dataset.isnull().sum())

# Numerical columns
imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')

imputer.fit(X[:, [0,3,4,5,6,7,9]])

X[:, [0,3,4,5,6,7,9]] = imputer.transform(X[:, [0,3,4,5,6,7,9]])

print("\nMissing Values Handled Successfully")


## STEP 4: Encoding categorical data

# One Hot Encoding for Geography

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [1])
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

# Label Encoding for Gender

le = LabelEncoder()

X[:,4] = le.fit_transform(X[:,4])

print("\nCategorical Data Encoded Successfully")

## STEP 5: Normalizing the data

sc = StandardScaler()

X = sc.fit_transform(X)

print("\nData Normalized Successfully")

## STEP 6: Splitting the dataset into
# Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset Split Completed")

print("\nTraining Feature Shape :", X_train.shape)
print("Testing Feature Shape :", X_test.shape)

print("\nTraining Target Shape :", y_train.shape)
print("Testing Target Shape :", y_test.shape)

print("\nFirst Five Rows of Processed Data")

print(pd.DataFrame(X_train).head())

print("\nData Preprocessing Completed Successfully")

~~~


## OUTPUT:

![alt text](<Screenshot 2026-07-30 162231-1.png>)
![alt text](<Screenshot 2026-07-30 162246-1.png>)
![alt text](<Screenshot 2026-07-30 162319-1.png>)
![alt text](<Screenshot 2026-07-30 162328-1.png>)
![alt text](<Screenshot 2026-07-30 162338-1.png>)
![alt text](<Screenshot 2026-07-30 162345-1.png>)
![alt text](<Screenshot 2026-07-30 162352-1.png>)
![alt text](<Screenshot 2026-07-30 162412-1.png>)

## RESULT:

Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


 S