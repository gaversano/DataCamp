# Intermediate Supervised  Learning in Python (Track)

LO1: Learners will be able to optimize model performance through feature engineering and hyperparameter tuning. 

LO2: Learners will be able to implement decision tree models.

LO3: Learners will be able to choose between neural network architectures (Perception, CNN, RNN) for common applications.

LO4: Learners will be able to implement ensemble models using voting, stacking, bagging and boosting.


### Courses

#### 1. Machine Learning Project Lifecycle 
    
    Some courses only focus on data preparation and building models. Discover the complete end to end machine learning project lifecycle. 
    Learn how to formulate your problem statement, differentiate between supervised and unsupervised learning as well as regression and classification problems, 
    train and comapre models and what to do after a model is deployed. 
    We will use real world datasets including Pokemon, wine quality and housing prices. This course will prepare you for success for your machine learning projects! 
    
    LO1: Learner will be able to order the steps of a machine learning project lifecycle.
        
    LO2: Learner will be able to create a histogram of features before/after deployment to assess risk of model drift.    
    
    
#### 2. Feature Engineering & Techniques
    
    Data is often messy and requires processing before you can use it in your models. 
    Transform your model inputs to put a smile on your model's face (and also your boss' face!). 
    Practice working with both continuous and categorical data using industry best practises.
    We will review techniques such as scaling, binning and one-hot encoding and how to create features from images and text features. 
    We will work with animal photos and text message datasets. Finally, we will learn techniques to handle missing data. 
    This course will give you the hands on experience you need to tackle your own unique datasets.    
    
    LO1: Learners will be able to apply scaling, binning and one-hot encoding using sklearn.preprocessing. 
    
    LO2: Learners will be able to transform a textual feature using TD-IDF.  
    
    LO3: Learners will be able to impute missing data using mean, median and mode using sklearn.impute.SimpleImputer.
    
    Prerequisites:
        
        - Supervised Learning with scikit-learn (Learners will implement linear regression and logistic models after feature engineering in the final chapter.)

#### 3. Hyperparameter Turning & Model Evaluation

    Not all models are made the same; tune your model and optimize performance! To pick the best model, we need to establish an evaluation metric. 
    We will review metrics for regression and classification problems and use them to choose the best hyperparameters for our model. 
    In this course you will optimize hyperparameters for linear regression, knn and SVM models using real world datasets. 
    We will use a brute force grid search of all possible combinations and also use a randomized search within a defined range. 
    Once you complete the course, try these techniques on your own model to improve performance.   
    
    LO1: Learners will be able to evaluate model performance of regression and classification problems using MSE, MAE, precision,  recall and F1 score. 
    
    LO2: Learners will be able to optimize model hyperparameters using GridSearchCV and RandomizedSearchCV. 
    
    Prerequisites:
    
        - Supervised Learning with scikit-learn (Learners will tune hyperparameters for linear regression and knn models.)
        
        - Linear Classifiers in Python (Learners will tune hyperparameters for linear svm models.)

#### 4. Decision Trees

    Decision trees are a useful class of models that can be used for both regression and classification problems often with little data processing. 
    They are especially useful when explaining how the model made it's prediction. 
    In this course, we will learn how decision trees are trained and use a decision tree to predict the genre of a song based on its audio features! 
    We will then inspect the tree to understand how it made the predictions. Try using a decision tree on your own model. 
    How does the performance stack up against other models?  
    
    LO1: Learners will be able to calculate the Gini of a decision tree node for a multiclass feature.
    
    LO2: Learners will predict the song genre using the DecisionTreeClassifier.  
    
    LO3: Learners will interpret a decision tree to explain how it made the classification decision.  
    
     Prerequisites:
    
        - Supervised Learning with scikit-learn (Learners should be familiar with regression and classification problems.)

#### 5. Neutral Networks & Deep Learning

    Neutral networks and deep learning have taken the world by storm powering some of the most impressive models including Netflix's recommendation system. 
    It can be challenging to navigate this dynamic field. In this course, we will use the cutting edge Keras library to build neural networks to predict the rating of apps in the Google Play Store. 
    Then, we will get introduced to some more advanced architectures: CNNs and RNNs. We will make use of these models by building an image classifier and predicting the next word from Tweets. 
    This course will help you decide which architecture is right for your project. 
    
    LO1: Learners will be able to predict a continuous target using a Keras neural network.  
    
    LO2: Learners will be able to choose between Perception, CNN and RNN architectures for different problem statements.
    
    Prerequisites:
    
    - Supervised Learning with scikit-learn (Learners should be familiar with regression and classification problems.)
    - Introduction to Deep Learning in Python (Learners should be familiar with theory behind NNs. This course will focus on application of different NNs.)
    
    
#### 6. Ensemble Models

    With so many models, how should we pick the best one for production?. They say that two heads are better than one and the same is true for machine learning models too! Ensembling is a technique that combines multiple models to create a super model. Oh and did we mention that that pros often use ensembling to win Kaggle competitions!? We will use voting, bagging, boosting and stacking to bring together our models to improve performance. We will predict movie revenue using data from TMDb and predict if a Pokemon is Legendary. Try these techniques on your own models to bring your performance to new heights. 
    
    LO1: Learners will be able to define voting, bagging, boosting and stacking.
    
    LO2: Learners will be able to implement a bagging regressor using BaggingRegressor and boosting regressor using GradientBoostingRegressor. 
    
    LO3: Learners will be able to optimize the hyperparameters of ensemble models.  
    
    Prerequisites:
    
    - Supervised Learning with scikit-learn (Learners should be familiar with regression and classification problems as well as hyperparameter tuning.)
    
