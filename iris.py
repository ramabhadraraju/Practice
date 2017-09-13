#Simple iris dataset classification using Knn and logistic regression.
#Load datasets
from sklearn.datasets import load_iris
iris= load_iris()

#separate features and labels
x= iris.data
y= iris.target

#Split data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(x_train,y_train)
z=cls.predict(x_test)
#accuracy prediction
from sklearn import metrics
print(metrics.accuracy_score(y_test,z))

# knn for different depths
from sklearn.neighbors import KNeighborsClassifier
scores=[]
k_range=range(1,26)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    # import metrics to calculate accuracy
    from sklearn import metrics
    accuracy=metrics.accuracy_score(y_test, knn_pred)
    scores.append(accuracy)
print(scores)
#display the best k using matplotlib
import matplotlib.pyplot as plt
plt.plot(k_range,scores)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()




