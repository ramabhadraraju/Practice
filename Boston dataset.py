import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn import grid_search
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score

# load boston dataset
dataset = load_boston()

# capture target and predictors
target = dataset.target
predictors = dataset.data
train_err_list = []
test_err_list = []
training_size = []
X, Y = predictors, target
depth = [1, 3, 5, 10]
split = np.arange(0.1, 0.9, 0.05)
print(split)

# calculate train and test errors for different test sizes and different depths
for depths in depth:
    for splits in split:
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=splits, random_state=99)
        regressor = DecisionTreeRegressor(max_depth=depths)
        regressor.fit(X_train, Y_train)
        train_err = metrics.mean_squared_error(Y_train, regressor.predict(X_train))
        test_err = metrics.mean_squared_error(Y_test, regressor.predict(X_test))
        train_err_list.append(train_err)
        test_err_list.append(test_err)
        training_size.append(len(X_train))
    plt.figure()
    plt.title('DecisionTreeRegressor: ERROR vs TRAIN SIZE')
    plt.plot(training_size, test_err_list, lw=2, label='test error')
    plt.plot(training_size, train_err_list, lw=2, label='train error')
    plt.xlabel('Training Size')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    train_err_list[:] = []
    test_err_list[:] = []
    training_size[:] = []

# performance of model for varying depths of tree
depth_arr = []
for tdepth in range(1, 13):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=99)
    regressor = DecisionTreeRegressor(max_depth=tdepth)
    regressor.fit(X_train, Y_train)
    train_err = metrics.mean_squared_error(Y_train, regressor.predict(X_train))
    test_err = metrics.mean_squared_error(Y_test, regressor.predict(X_test))
    train_err_list.append(train_err)
    test_err_list.append(test_err)
    depth_arr.append(tdepth)
plt.figure()
plt.title('DecisionTreeRegressor: ERROR vs DEPTH')
plt.plot(depth_arr, test_err_list, lw=2, label='test error')
plt.plot(depth_arr, train_err_list, lw=2, label='train error')
plt.xlabel('DEPTH')
plt.ylabel('MSE')
plt.legend()
plt.show()

# predicting the best depth and value of housing price for unseen data

# parameters range for decisiontree regressor
grid = {'max_depth': list(range(1, 15)), 'splitter': ['best', 'random'],
        'max_features': list(range(1, 14))}
# grid search to identify the optimal parameter values
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(DecisionTreeRegressor(random_state=0), grid, refit=True, cv=15)
search.fit(predictors, target)
print("Best param:", search.best_params_)
predict_new = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
price = search.predict(predict_new)
print("price" + str(price))