import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn

data=pd.read_csv("C:\\Users\exam\Desktop\Python SBS final\loan prediction.csv")

data.head(5)
data.dtypes
data["LoanAmount"].describe()
d1=data["LoanAmount"].groupby(data["Education"]).describe()
d1

data["LoanAmount"].fillna(0,inplace=True)

# histogram
data["LoanAmount"].plot.hist(bins=50)
# histogram using seaborn
#For both histo and kde/density curve
sea.distplot(data["LoanAmount"],bins=50)
sea.distplot(data["LoanAmount"],bins=50, kde=False)
sea.distplot(data["LoanAmount"],bins=50, hist=False)
#counting individual numbers from the dependent column
aa=data['Dependents'].value_counts()
aa
aa.plot.bar()
sea.barplot(x=aa.index,y=aa.values)

#Cross tab education and self employed (Stacked chart)
bb=pd.crosstab(data["Education"],data["Self_Employed"])
bb
bb.plot.bar(stacked=True)
plt.tight_layout()

#Correlation
data[['ApplicantIncome','LoanAmount']].corr()
#data['ApplicantIncome'].corr(data['LoanAmount])

sea.regplot(data['ApplicantIncome'],data['LoanAmount'])
plt.clf()

data1=data[data['ApplicantIncome']<30000]
sea.regplot(data1['ApplicantIncome'],data1['LoanAmount'])

sea.barplot(x='Education',y='LoanAmount',data=data)

data.boxplot(column='LoanAmount',by='Married')
sea.boxplot(x='Married',y='LoanAmount',data=data)
sea.boxplot(x='Married',y='LoanAmount',data=data,hue='Education')
#sea.swarmplot(x='Married',y='LoanAmount',data=data)

sea.pairplot(data[['LoanAmount','ApplicantIncome','CoapplicantIncome']])

