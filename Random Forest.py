import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=3,min_samples_leaf=5,random_state=100)

#classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=3,min_samples_leaf=5,random_state=100)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

print("Training Accuracy:{:.3f}".format(classifier.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(classifier.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(classifier,X,y,cv=5)
#accuracies=cross_val_score(log_reg,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean())) #There is no need of data scaling here as in this model there is no comparison
# of one column to another as it is forming branches within its columns(or features), even if it is scaled than the final answer if visualized
# in the tree(decision tree), it will show the Z values

#Gridsearch (it is used to find the best estimator,leafs,max depth etc.)
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators': [10,20,30],'max_depth':[3,5,7],'min_samples_leaf': [2,5,10]}

cv_rfc=GridSearchCV(estimator=classifier,param_grid=param_grid,cv=5)
cv_rfc.fit(X_train,y_train)
print(cv_rfc.best_score_.round(5))
y_pred=cv_rfc.predict(X_test)

print('Testing Accuracy:{:.3f}'.format(cv_rfc.score(X_test,y_test)))

print(cv_rfc.best_params_)

#Feature importance (which feature, i.e. column is the most important in this model)
n_features=df.shape[1]
plt.barh(range(n_features),classifier.feature_importances_,align='center')
plt.yticks(np.arange(n_features),df.columns)
plt.xlabel('Feature Importance')
plt.tight_layout()

imp=list(zip(np.round(classifier.feature_importances_,2),df.columns))
imp.sort(reverse=True)
print(imp)