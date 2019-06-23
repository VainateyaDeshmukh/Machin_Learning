
import pandas as pd
import pydotplus as pdtp
from io import StringIO
from sklearn import tree

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
#df=data.drop(data[["Loan_ID","Loan_Status","Loan_Amount_Term"]],axis=1)
df=data.iloc[:,1:-1]
df=pd.get_dummies(df,columns=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"])

X=df.values
y=data.iloc[:,-1].values

#Split the data into a model of 25% and 75%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

clf_gini=DecisionTreeClassifier(criterion="gini",random_state=10,max_depth=3,min_samples_leaf=5)
clf_gini.fit(X_train,y_train)
y_pred=clf_gini.predict(X_test)

print("Training Accuracy:{:.3f}".format(clf_gini.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(clf_gini.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

len2=len(df.columns)
features=list(df.columns[0:len2])

dot_data= StringIO();
tree.export_graphviz(clf_gini,out_file=dot_data, feature_names=features,impurity=False)
graph=pdtp.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('tdemo.pdf')

####

df1=pd.DataFrame(X_train,columns=df.columns)
df1['Loan_Status']=y_train
df1.to_csv('Train.csv')