import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("D:\Analytics (sem3)\Machine Learning\loan prediction.csv")

#null
data1=data.copy()

data1.isnull().sum()
len(data1)
#drop rows
data1.dropna(inplace=True)
len(data1)#as the length of the droped na is very less i.e 480 as compared to the original 614, which is almost the 20-25% of the data so this cannot be carried on hence we only delete a particular column as shown below

data1=data.copy()
data1.dropna(subset=['LoanAmount'],inplace=True)

#Fill with 0
#data1.fillna(0,inplace=True)
data1['LoanAmount'].fillna(0,inplace=True)

#mean (146)
data1=data.copy()
data1.dropna(subset=['LoanAmount'],inplace=True)

data1['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
#median (128)
data1=data.copy()
data1.dropna(subset=['LoanAmount'],inplace=True)

data1['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)
#median is better than mean in this data as the data is skewed towards one side that was visualized earlier

#Mode
data1=data.copy()
data1.dropna(subset=['LoanAmount'],inplace=True)
data['Self_Employed'].value_counts()/len(data['Self_Employed'])
data1['Self_Employed'].fillna('No',inplace=True)
mode1=data1['Self_Employed'].mode()
data1['Self_Employed'].fillna(mode1[0],inplace=True)

#Feature Engineering/Extraction
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']

#Feature Scaling
from sklearn.preprocessing import StandardScaler
data1=data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]

data_scale=data1.copy()
data_scale.fillna(0,inplace=True)

#z=(x-mean)/std dev
sc=StandardScaler()
data2=sc.fit_transform(data_scale) # For example if In case of age and salary,numbers are very different from each other to compare them,
# hence we use this to covert it into the z scale so as to compare them on an equal scale

#Label Encoding
from sklearn.preprocessing import LabelEncoder
data1=data.copy()
data1['Married'].value_counts()
mode1=data1['Married'].mode()
data1['Married'].fillna(mode1[0],inplace=True)
le=LabelEncoder()
data1['Married']=le.fit_transform(data1['Married']) #converts No & Yes into 0 & 1 in the data set


#Dummy variables (One hot encoder)
dummies1=pd.get_dummies(data,columns=['Education','Dependents'])