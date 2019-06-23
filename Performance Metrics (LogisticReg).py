import pandas as pd
from sklearn.metrics import*
import matplotlib.pyplot as plt
data=pd.read_csv("C:\Pyproject\ML\loan prediction.csv")

data.dtypes

data.isnull().sum()
data['Gender'].fillna(data["Gender"].mode()[0],inplace=True)
data['Married'].fillna(data["Married"].mode()[0],inplace=True)
data['Dependents'].fillna(data["Dependents"].mode()[0],inplace=True)
data['Self_Employed'].fillna(data["Self_Employed"].mode()[0],inplace=True)
data['Loan_Amount_Term'].fillna(data["Loan_Amount_Term"].mode()[0],inplace=True)
data['Credit_History'].fillna(data["Credit_History"].mode()[0],inplace=True)
data['LoanAmount'].fillna(data["LoanAmount"].median(),inplace=True)

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

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)

print("Training Accuracy:{:.3f}".format(log_reg.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(log_reg.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(log_reg,X,y,cv=10)
#accuracies=cross_val_score(log_reg,X,y,cv=5)
print('{:.3f}'.format(accuracies.mean())) #This validation is without the standardised values i.e. it is on the original values that we don't want

#Pipeline (This is used to make space for standardization and log_reg togather in one variable and use it togather on validation)
from sklearn.pipeline import make_pipeline
clf=make_pipeline(sc,log_reg) # Variable(clf) assigned to store sc(standard scale) and log_reg (logarithmic regression) function
accuracies=cross_val_score(clf,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))

#Probability
from sklearn.metrics import*
predicted_proba=log_reg.predict_proba(X_test)
y_test1=(y_test=='Y').astype('int')

#ROC curve
fpr,tpr,thresholds=roc_curve(y_test1,predicted_proba[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.rcParams['front.size']=12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1- Specificity')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

auc=roc_auc_score(y_test1,predicted_proba[:,1])
print('AUC:%.2f'%auc)