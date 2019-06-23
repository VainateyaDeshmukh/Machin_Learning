import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:\Pyproject\ML\loan prediction.csv")

#Feature Engineering
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']

#To create a new data set of only
data1=data[['TotalIncome','LoanAmount']]

#To check the null values
data1.isnull().sum()
#Fill the null values
data1['LoanAmount'].fillna(data1['LoanAmount'].median(),inplace=True)
#Recheck if there are still any null values
data1.isnull().sum()
#Putting values of data1 into variable X
X=data1.values
#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Cutomers')
plt.ylabel('Euclidian Distances')
plt.show()

#Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()