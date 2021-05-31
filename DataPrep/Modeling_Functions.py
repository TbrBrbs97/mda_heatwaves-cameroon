#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import functions
#Packages
import numpy as np
import sys
import pandas as pd
import wbgapi as wb
import sklearn.preprocessing
import seaborn as sns
from pandas import DataFrame
from scipy.stats import shapiro
from sklearn.covariance import MinCovDet
import sys
import os
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import scipy as scy
import pandas as pd
from time import time
#from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, PoissonRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sb
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


class Scale_predictor_variables:
    def __init__(self,X_train,y_train,pre_process,cv):
        self.X_train = X_train
        self.y_train = y_train
        self.pre_process = pre_process
        self.cv = cv
    def Plot_cross_validation_results(self):
        pca = PCA()

        X_reduced = pca.fit_transform(self.pre_process.fit_transform(self.X_train))
                                

        #define cross validation method
        #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

        regr = LinearRegression()
        mse = []

        # Calculate MSE with only the intercept
        score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), self.y_train, cv=self.cv,
           scoring='neg_mean_squared_error').mean()    
        mse.append(score)

        # Calculate MSE using cross-validation, adding one component at a time
        for i in np.arange(1, 6):
            score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], self.y_train, cv=self.cv, scoring='neg_mean_squared_error').mean()
            mse.append(score)
    
        # Plot cross-validation results    
        plt.plot(mse)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('MSE')
        plt.title('CDD')
        variance_ratio = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        print(variance_ratio)
        
            


# In[38]:


class Scoring:
    def __init__(self,pre_process,X_train,y_train):
        self.pre_process = pre_process
        self.X_train = X_train
        self.y_train = y_train
    def Score(self):
        ## data 
        ########################################################################
        model_1 = RandomForestRegressor(max_depth=15,random_state=0)
        model_2 = LinearRegression(fit_intercept=True)
        model_3 = Ridge(alpha=5)
        model_4 = Lasso(alpha=10)
        model_5 = SVR(C=2.5, epsilon=0.5)
        model_6 = GradientBoostingRegressor(random_state=0)
        model_7 = PoissonRegressor()
        
        

        MSE = []
        R2 = []
        for mymodels in [model_1,model_2,model_3,model_4,model_5,model_6,model_7]:
            model_pipeline = Pipeline(steps=[('pre_processing',self.pre_process),('scaler', StandardScaler()),('reduce_dim', PCA()),
                                 ('model', mymodels)
                                 ])
            model_pipeline.fit(self.X_train,self.y_train)
            MSE.append(mean_squared_error(self.y_train,model_pipeline.predict(self.X_train))**0.5)
            R2.append(r2_score(self.y_train,model_pipeline.predict(self.X_train)))
    
        print(np.round(MSE,2))   
        print(np.round(R2,2))


# In[43]:


class Model_select:
    def __init__(self,X_train,y_train,X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def model_selection(self):
        """
        hyperparameter tuning is performed using GridSearchCV
        technique uses cross-validation when applying the default values of a 5-fold cross validation 
        as a means of splitting the training data into a training and validation sets.
        model score is representen with the R-squared metrics
        """
        models = []
        models_1 = ["Ridge","Lasso","LinearRegression","PoissonRegressor"]
        models_2 = ["RandomForestRegressor","GradientBoostingRegressor"]
        model_3 = ["SVR"]
        models += models_1 + models_2 + model_3
        models_dictionary = {"Ridge":Ridge(),"Lasso":Lasso(),"LinearRegression":LinearRegression(fit_intercept=True),
                             "RandomForestRegressor":RandomForestRegressor(random_state=0),"GradientBoostingRegressor":GradientBoostingRegressor(random_state=0),
                            "SVR":SVR(epsilon=0.5),"PoissonRegressor":PoissonRegressor(max_iter=200)}
        models_score = {}
        
        
        # Tuning of parameters for regression by cross-validation
                    # Number of cross valiations is 5
        
        for model in models:
            if model in models_1:
                
                pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reduce_dim', PCA()),
                ('regressor', models_dictionary[model])
                ])
                pipe = pipe.fit(self.X_train, self.y_train)
                n_features_to_test = np.arange(1, 13)
                alpha_to_test = 2.0**np.arange(-6, +6)
            
                if model == "LinearRegression":
                    params = {'reduce_dim__n_components': n_features_to_test,
                    'scaler' : [StandardScaler(), RobustScaler()]}
                else:
                    params = {'reduce_dim__n_components': n_features_to_test,
                    'regressor__alpha': alpha_to_test,
                    'scaler' : [StandardScaler(), RobustScaler()]}
                gridsearch = GridSearchCV(pipe, params, verbose=1).fit(self.X_train, self.y_train)
                
            elif model in models_2:
                
                if model == "RandomForestRegressor":
                  
                    
                    model_estimator =models_dictionary[model]
                    params={'n_estimators':[20,30,40,60,80,100], 'max_depth': 
                    [5,10,15,20],'max_features':[2,5,8]}
                    
                     
                else:
                    model_estimator =  models_dictionary[model]
                    
                    params = {'learning_rate': [0.01,0.02,0.03,0.04],
                    'subsample'    : [0.9, 0.5, 0.2, 0.1],
                    'n_estimators' : [20,30,40,60,80,100],
                    'max_depth'    : [4,6,8,10]
                     }
                
                gridsearch = GridSearchCV(estimator = model_estimator,param_grid = params,n_jobs=-1).fit(self.X_train, self.y_train)
            else:
                parameters = {'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 2.5, 5,7.5,10,15]}
                gridsearch = GridSearchCV(models_dictionary[model], parameters).fit(self.X_train, self.y_train)
             
            print(" Results from Grid Search:",model)
            print("\n The best estimator across ALL searched params:\n",gridsearch.best_estimator_)
            print("\n The best score across ALL searched params:\n",gridsearch.best_score_)
            print("\n The best parameters across ALL searched params:\n",gridsearch.best_params_)
            print('\n Final score is: ', gridsearch.score(self.X_test, self.y_test))
            print("")
            models_score[model] = gridsearch.score(self.X_test, self.y_test)
        self.models_score = models_score

        


# In[ ]:




