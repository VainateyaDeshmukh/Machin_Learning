from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

data=pd.read_csv("C:\\Users\exam.SBS\PycharmProjects\ML2\loan prediction.csv")

data.dtypes

data.isnull().sum()
data["Gender"].fillna(data["Gender"].mode()[0],inplace=True)
data["Married"].fillna(data["Married"].mode()[0],inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0],inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0],inplace=True)
data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)
#data["Loan_Amount_Term"].value_counts()
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0],inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0],inplace=True)
data.isnull().sum()

#Excluding/dropping non value added columns or data
df=data.drop(data[["Loan_ID","Loan_Status","Loan_Amount_Term"]],axis=1)
#df=data.iloc[:,1:-1]
df=pd.get_dummies(df,columns=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"])

X=df.values
y=data.iloc[:,-1].values

#Split the data into a model of 25% and 75%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier

Knn=KNeighborsClassifier(n_neighbors=5) #Hyperparameter= whose value is set before the learning process begins
Knn.fit(X_train,y_train)
y_pred=Knn.predict(X_test)


print("Training Accuracy:{:.3f}".format(Knn.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(Knn.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(Knn,X,y,cv=10)
#accuracies=cross_val_score(log_reg,X,y,cv=5)
print('{:.3f}'.format(accuracies.mean())) #This validation is without the standardised values i.e. it is on the original values that we don't want

#Pipeline (This is used to make space for standardization and log_reg togather in one variable and use it togather on validation)
from sklearn.pipeline import make_pipeline
clf=make_pipeline(sc,Knn) # Variable(clf) assigned to store sc(standard scale) and Knn (Neighbors Classifier) function
accuracies=cross_val_score(clf,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))

#To find out the best K value i.e. n, for the better accuracy

neighbors=range(1,11)
k_score=[]
for n in neighbors:
    knn1=KNeighborsClassifier(n_neighbors=n)
    clf1=make_pipeline(sc,knn1)
    accuracies1=cross_val_score(clf1,X,y,cv=10)
    k_score.append(accuracies1.mean())
    print('{:.3f}'.format(accuracies1.mean())) #print the # of accuracies

import matplotlib.pyplot as plt
plt.plot(neighbors,k_score)
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
