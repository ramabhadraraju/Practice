#Import pandas
import pandas as pd
data=pd.read_csv('C:\\Users\Raju garu\Desktop\Advertising.csv',index_col=0)
print(data.head())
#import seaborn to visualize
import seaborn as sns
import matplotlib.pyplot as plt
#using scatter plots
plot = sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',size=7,aspect=0.7,kind='reg')
plot.savefig('C:\\Users\Raju garu\Desktop\scatterplot')
feature_cols=['TV','radio','newspaper']
X=data[feature_cols]
print(X.head())
y=data['sales']
#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print(lr.intercept_)
print(lr.coef_)
print(list(zip(feature_cols,lr.coef_)))
#Calculate rmse for linear regression
import numpy as np
from sklearn import metrics
rmse= np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print(rmse)
