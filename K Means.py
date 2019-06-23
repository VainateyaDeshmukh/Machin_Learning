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

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
list1=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X)
    list1.append(kmeans.inertia_) #inertia= WCSS
plt.plot(range(1,11),list1,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  #WCSS= Within cluster sum of squares

#Fitting K-means to the dataset
kmeans=KMeans(n_clusters=5,random_state=10)
y_kmeans=kmeans.fit_predict(X)

data['kmeans']=y_kmeans
data.replace({'kmeans':{0:'Red',1:'Blue',2:'Green',3:'cyan',4:'magenta'}})
data["kmeans"].value_counts()

#Visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.legend()

kmeans.cluster_centers_