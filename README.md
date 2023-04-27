# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```

import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```

```
Titanic.csv :

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

# removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

# data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

# feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```


# OUPUT
## Data.csv :
## Initial Dataset:
![dsex51](https://user-images.githubusercontent.com/120623583/234909040-174710df-273f-49fe-bb0c-4475d925f929.png)

## Binary Encoding:
![dsex52](https://user-images.githubusercontent.com/120623583/234909103-f13449d0-7b2b-452d-a690-314b0bd33a8f.png)

![dsex53](https://user-images.githubusercontent.com/120623583/234909168-69f091dc-5d27-44ee-a7bb-024b1363bfc7.png)

## Encoded Dataset:
![dsex54](https://user-images.githubusercontent.com/120623583/234909236-ee6f460f-20b4-423d-b37f-f7ff11864d29.png)

## Data Scaling using MinMaxScaler:
![dsex55](https://user-images.githubusercontent.com/120623583/234909345-e2bccad0-d3c4-41b8-9ba3-374444f03eeb.png)

##Data Scaling using StandardScaler:
![ex56](https://user-images.githubusercontent.com/120623583/234909763-fd742473-29da-450c-b48f-741fc88f5321.png)

## Data Scaling using MaxAbsScaler:
![dsex57](https://user-images.githubusercontent.com/120623583/234909808-e8cabb01-0022-4386-a1ad-4d5ace2aa426.png)]

## Data Scaling using RobustScaler:
![dsex58](https://user-images.githubusercontent.com/120623583/234909852-ab188a56-a015-430a-954e-80f30c7a94b3.png)

## Encoding.csv :
## Initial Dataset:
![dsex59](https://user-images.githubusercontent.com/120623583/234909902-5c9fbac2-3288-48b9-8ecc-bb86bd79c39d.png)

## Binary Encoding:
![dsex510](https://user-images.githubusercontent.com/120623583/234910035-39081e9f-0264-4b45-b811-af26d35a18be.png)
![dsex511](https://user-images.githubusercontent.com/120623583/234910102-02737c14-6c93-4268-b3cc-3a82140bb316.png)

## Encoded Dataset:
![dsex512](https://user-images.githubusercontent.com/120623583/234910125-89d2fbc2-0467-41a9-968f-543a0ae3cb08.png)
![dsex513](https://user-images.githubusercontent.com/120623583/234910165-7b20fa62-97ff-4464-887d-567c984f2dc3.png)

## Data Scaling using MinMaxScaler:
![dsex514](https://user-images.githubusercontent.com/120623583/234910193-5332abfe-8910-49dc-bee6-21c863c7e5b1.png)

## Data Scaling using StandardScaler:
![dsex515](https://user-images.githubusercontent.com/120623583/234910250-8c479cbe-9659-430d-bed5-21650c0726c6.png)

## Data Scaling using MaxAbsScaler:
![dsex516](https://user-images.githubusercontent.com/120623583/234910334-206ea55b-e804-4b78-8e10-f17b8877483b.png)

## Data Scaling using RobustScaler:
![dsex517](https://user-images.githubusercontent.com/120623583/234910389-fb6b888c-54bd-4ad0-899e-1325e624e817.png)

## Titanic.csv : Initial Dataset:
![dsex518](https://user-images.githubusercontent.com/120623583/234910440-2f5260aa-0b91-48ca-b7eb-c95080501641.png)

## Data cleaning before encoding:
![dsex519](https://user-images.githubusercontent.com/120623583/234910490-a6d9ffc5-265d-4055-9536-d857f1992b8b.png)
![dsex520](https://user-images.githubusercontent.com/120623583/234910544-1a9b5dc4-e5c9-4f23-bebc-853669ccf083.png)

## Cleaned Dataset:
![dsex521](https://user-images.githubusercontent.com/120623583/234910598-2f4a1671-1b49-45aa-b36f-c8eac6cbbbcb.png)
  
## Binary Encoding:    
![dsex522](https://user-images.githubusercontent.com/120623583/234910636-15b0e807-cfff-4a76-9fa5-472e5dbf4154.png)

## Encoded Dataset:
![dsex523](https://user-images.githubusercontent.com/120623583/234910671-cb5c66c7-0590-4239-9d89-1ce71b4123b0.png)

## Data Scaling using MinMaxScaler:
![dsex524](https://user-images.githubusercontent.com/120623583/234910700-cc08fb7c-3c9b-4044-ba4c-c6ea306de978.png)

## Data Scaling using StandardScaler:
![dsex525](https://user-images.githubusercontent.com/120623583/234910781-cc183372-8249-4f06-bcf8-5eb018ecdcb0.png)

## Data Scaling using RobustScaler:
![dsex526](https://user-images.githubusercontent.com/120623583/234910831-9c40b140-1450-4c8c-87bc-976a59c215e2.png)

## RESULT:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.





