
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

from sklearn.svm import SVC
svm=SVC(kernel='linear')

svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Training Accuracy:{:.3f}".format(svm.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(svm.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


#Pipeline (This is used to make space for standardization and log_reg togather in one variable and use it togather on validation)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
clf=make_pipeline(sc,svm) # Variable(clf) assigned to store sc(standard scale) and log_reg (logarithmic regression) function
accuracies=cross_val_score(clf,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))
