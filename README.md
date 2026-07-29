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


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
STEP 2: Importing the dataset
STEP 3: Taking care of missing data
dataset = pd.read_csv("WineQuality.csv")
print(dataset.head())
STEP 3: Taking care of missing data

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)


## STEP 4: Encoding categorical data

if dataset.iloc[:, -1].dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

## STEP 5: Normalizing the data
sc = StandardScaler()
X = sc.fit_transform(X)

## STEP 6: Splitting the data into Train and Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

~~~


## OUTPUT:
![alt text](<Screenshot 2026-07-29 142125.png>)
 ![alt text](<Screenshot 2026-07-29 142057.png>)
  ![alt text](<Screenshot 2026-07-29 142102.png>) 
  ![alt text](<Screenshot 2026-07-29 142109.png>) 
  ![alt text](<Screenshot 2026-07-29 142113.png>) 
  ![alt text](<Screenshot 2026-07-29 142118.png>)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


