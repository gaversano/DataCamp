### Bagging Regressor Model (Exercise)


#### Context

Our weak linear regression model poorly predicts movie revenue. 
Let's train multiple linear regression models and put them together using the bagging technique.


#### Instructions

- Train a bagging regression model with 100 estimators. The base estimator linear model is loaded as 'lr'. The training set is loaded as 'X_train' and 'y_train'.

- Print the test score. The test set is loaded as 'X_test' and 'y_test'.


#### Code 

#import the bagging regression model

from sklearn.ensemble import _ _ _

#Build the bagging regression model
#The base estimator is loaded in variable lr

breg = _ _ _(_ _ _, n_estimators=_ _ _, random_state=42)

#Train the model, the training set is loaded in variables X_train and y_train

breg.fit(_ _ _, _ _ _)

#Print the score, the test set is loaded in variables X_test and y_test

print(breg._ _ _(X_test, y_test))


#### Solution 

#import the bagging regression model

from sklearn.ensemble import BaggingRegressor

#Build the bagging regression model
#The base estimator is loaded in variable lr

breg = BaggingRegressor(lr, n_estimators=100, random_state=42)

#Train the model, the training set is loaded in variables X_train and y_train

breg.fit(X_train, y_train)

#Print the score, the test set is loaded in variables X_test and y_test

print(breg.score(X_test, y_test))

