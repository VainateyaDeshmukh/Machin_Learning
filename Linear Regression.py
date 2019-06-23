import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("D:\Analytics (sem3)\Machine Learning\RegData.csv")

X=data.iloc[:,:-1].values #universal
y=data.iloc[:,1].values
#y=data.iloc[:,-1].values  #universal

#Split the data into a model of 25% and 75%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)

from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

#To check how good the model is:

#Coefficient of determination R^
print('Accuracy on training data:{:.3f}'.format(reg.score(X_train,y_train)))
print('Accuracy on test data: {:.3f}'.format(reg.score(X_test,y_test)))
#print('R2 Score {:.3f}'.format(r2_score(y_test,y_pred)))

#To find the coefficient and the intercept:
print(reg.coef_)
print(reg.intercept_)

#Visualization on a scatter plot:

plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Insurance Estimate - Training')
plt.xlabel('Income')
plt.ylabel('Insurance')


plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Insurance Estimate - Test')
plt.xlabel('Income')
plt.ylabel('Insurance')