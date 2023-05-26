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
DEVELOPED BY:SWETHA P
REGISTER NO:212222100053

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

#Dataset-1 (data.csv)
![Screenshot 2023-05-26 222140](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/278a7a94-4a72-46d6-8747-dfccc95fa230)
![Screenshot 2023-05-26 222231](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/2d97950b-f03b-499b-a7e7-22f6c1779ab3)
![Screenshot 2023-05-26 222300](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/1697e0d1-8307-4a78-886a-a15a0f430108)
![Screenshot 2023-05-26 222414](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/1d34f151-f655-4940-a720-ce6a289ef093)
![Screenshot 2023-05-26 222429](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/63f089d8-e05b-4b78-8f59-6363152de704)
![Screenshot 2023-05-26 222445](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/29258501-937d-4390-8e36-fbd0993a76c2)

#Dataset-2 (Encoding data.csv)
![Screenshot 2023-05-26 222719](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/4a72e3a5-0121-449e-985d-2cd2440ad5f4)
![Screenshot 2023-05-26 222734](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/30484bdb-16eb-4e04-9867-24ed3a72d7b6)
![Screenshot 2023-05-26 222750](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/791e6685-b42d-4178-8535-3f4ce0562727)
![Screenshot 2023-05-26 222807](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/a5de5449-77c6-4200-92ee-e768a4ba917e)
![Screenshot 2023-05-26 222846](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/e767932d-c2ac-492c-8953-ff3609f12bf7)
![Screenshot 2023-05-26 222905](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/59e328db-f331-4f33-a110-4e5f84d066ac)
![Screenshot 2023-05-26 222926](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/c0a025d3-02ac-4191-9723-2e3a03a999bd)
![Screenshot 2023-05-26 222942](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/e3988672-7650-4d55-83a8-797bbb1e2e2e)

#Dataset-3 (titanic_dataset.csv)
![Screenshot 2023-05-26 223045](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/a15d3cea-0431-4625-a7a8-ca8811410795)
![Screenshot 2023-05-26 223105](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/19d4e3b9-8e6b-4f08-8a35-1514bc1d4b31)
![Screenshot 2023-05-26 223213](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/09af1e34-7aa1-4776-a053-261c814aceb6)
![Screenshot 2023-05-26 223226](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/eaee60c0-5973-4191-b977-1106efe8713a)
![Screenshot 2023-05-26 223246](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/a6ac90d9-c356-4ea4-95ca-6221c5cf9ae8)
![Screenshot 2023-05-26 223319](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/2888cfeb-2a1b-4e9d-b52a-b47c5f56ebb8)
![Screenshot 2023-05-26 223340](https://github.com/swetha1510/EX-05-Feature-Generation/assets/120623583/16110702-ecb6-4e9c-8ba0-b1c86ff64604)


## RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.





