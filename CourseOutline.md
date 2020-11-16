# Ensemble Models (Course)

### Chapters 

1. **Chapter 1: Power of the Crowd - Combining Models**

    1.1. Lesson 1: Wisdom of the crowd
    
        LO1: Learners will be able to discover the motivations behind ensemble models.
        
        LO2: Learners will be able to estimate the better model based on the number of estimators.  

    1.2. Lesson 2: Voting 
    
        LO1: Learners will be able to define voting in regards to using the most often or highest score of several models.
        
        LO2: Learners will be able to implement hard voting from three models using the Counter class from Collections module. 
    
        LO3: Learners will be able to implement hard voting using the VotingRegressor model with three predictor models. 

        LO4: Learners will be able to recognize hard voting using the VotingClassifier model. 

    1.3. Lesson 3: Stacking 
    
        LO1: Learners will be able to define stacking in regards to using output of several models as input to a final estimator.
    
        LO2: Learners will be able to predict Pokemon Legendary status using the StackingClassifier model with a Logistic Regression estimator with three predictor models. 
        
        LO3: Learners will be able to recognize stacking using the StackingRegressor model. 
    

2. **Chapter 2: Bagging Ensemble Methods **

    2.1. Lesson 1: What is Bagging? 
    
        LO1: Learners will be able to define bagging in regards to utilizing the same predictor estimator and random sampling with replacement of the training set.

    2.2. Lesson 2: BaggingRegressor Model
        
        LO1: Learners will be able to fit a BaggingRegressor model and predict movie revenue.
        
        LO2: Learners will be able to recognize bagging using the BaggingClassifier model. 

    2.3. Lesson 3: Random Forest Model
    
        LO1: Learners will be able to define Random Forest in regards to utilizing the decision tree estimator and random sampling features with replacement.
        
        LO2: Learners will be able to fit a RandomForestClassifier model and predict Pokemon legendary status.

        LO3: Learners will be able to recognize Random Forest using the RandomForestRegressor model. 

3. **Chapter 3: Boosting Ensemble Methods**

    3.1. Lesson 1: What is Boosting? 
    
        LO1: Learners will be able to define boosting in regards to utilizing the same predictor estimator and training sequentially to correct errors. 

    3.2. Lesson 2: AdaBoostClassifier
    
        LO1: Learner will be able to define Adaptive Boosting in regards to sequentially training using the same predictor estimator and focusing on training examples with high prediction error. 
        
        LO2: Learners will be able to fit a AdaBoostClassifier model and predict Pokemon Legendary status.
        
        LO3: Learners will be able to recognize boosting using the AdaBoostRegressor.

    3.3. Lesson 3: GradientBoostingRegressor
    
        LO1: Learner will be able to define Gradient Boosting in regards to sequentially training using the same predictor estimator but fitting to the residual error. 
        
        LO2: Learners will be able to fit a GradientBoostingRegressor model and predict movie revenue.

        LO3: Learners will be able to recognize boosting using the GradientBoostingRegressor.

4. **Chapter 4: Tunning and Comparing Ensemble Methods**

    4.1. Lesson 1: BaggingRegressor Hyperparameter Tuning
    
        LO1: Learners will be able to plot the test set error as a function of the number of predictors.
       
        LO2: Learners will be able to optimize hyperparameters of a BaggingRegressor model using GridSearchCV.
        

    4.2. Lesson 2: GradientBoostingRegressor Hyperparameter Tuning
    
        LO1: Learners will be able to plot the test set error as a function of the learning rate.
        
        LO2: Learners will be able to optimize hyperparameters of a GradientBoostingRegressor model using GridSearchCV.    

    4.3. Lesson 3: Comparing Models
        
        LO1: Learners will be able to compare models using MSE.
