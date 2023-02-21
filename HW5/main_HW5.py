import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

def regression_movies(r, alpha = 1.0):
    """
    Compute the r_2 score of the corresponding method

    Args:
        r - A indicated of selecting the method, which should be in [0, 1, 2].
        alpha - A parameter that is required to be used for Ridge regression and Lasso regression

    Returns: r_2 score
    """
    
    # Load data here from the pandas dataframe 
    df = pd.read_csv('./movies_clean.csv')
    #df = pd.DataFrame(dff)
    regression_target = 'revenue'
    all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                      'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy',
                      'Romance', 'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

   
    # Extract here the data you will use to fit the models
    X_train, X_test, y_train, y_test = train_test_split(df.all_covariates, df.regression_target) 

    # Now fit each of the models
    if r == 0:
        reg = linear_model.LinearRegression()
        
    elif r == 1:
        reg = linear_model.Ridge(alpha = alpha)
        
    elif r == 2:
        reg = linear_model.Lasso(alpha = alpha)
        
    else:
        print("r should be 0, 1, or 2")
        return 0
    x = X_train.reg
    return r2_score(df.regression_target,x)
# the corresponding r2score value



def classification_movies(r,  C = 1.0):
    """
    Compute the r_2 score of the corresponding method

    Args:
        r - A indicated of selecting the method, which should be in [1, 2].
        C - A parameter that is required to be used for SVM

    Returns: accuracy_score
    """
    # Load data
    df = pd.read_csv('./movies_clean.csv')

    classification_target = 'profitable'
    all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                      'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy',
                      'Romance', 'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

    # Extract here the data you will use to fit the models


    # Now fit each of the models
    
    if r == 1:
        reg = linear_model.LogisticRegression()
    elif r == 2:
        reg = svm.SVC(C = C)
    else:
        print("r should be  1, or 2")
        return 0

    return # the corresponding accuracy score

def refined_regression_movies(r, alpha = 1.0):
    """
    Compute the r_2 score of the corresponding method

    Args:
        r - A indicated of selecting the method, which should be in [0, 1, 2].
        alpha - A parameter that is required to be used for Ridge regression and Lasso regression

    Returns: r_2 score
    """
    # Load data
    df = pd.read_csv('./movies_clean.csv')
    regression_target = 'revenue'
    all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                      'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy',
                      'Romance', 'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

    # Define a new_df excluding or dropping data
    
    new_df = pd.read_csv('./movies_clean.csv', skiprows = to_exclude)

    # Find the new data set for fitting the models
  
    # Now fit each of the models
     
    if r == 0:
        reg = linear_model.LinearRegression()
    elif r == 1:
        reg = linear_model.Ridge(alpha = alpha)
    elif r == 2:
        reg = linear_model.Lasso(alpha = alpha)
    else:
        print("r should be 0, 1, or 2")
        return 0

    return  # the corresponding score

def refined_classification_movies(r, C = 1.0):
    """
    Compute the r_2 score of the corresponding method

    Args:
        r - A indicated of selecting the method, which should be in [0, 1, 2].
        n_neighbors - A parameter that is required to be used for KNN
        C - A parameter that is required to be used for SVM

    Returns: accuracy_score
    """
    # Load data
    df = pd.read_csv('./movies_clean.csv')

    # Define here a new_df with excluding or dropping data

    # Find the new data set for fitting the models

    # Now fit each of the models
    
    if r == 1:
        reg = linear_model.LogisticRegression()
    elif r == 2:
        reg = svm.SVC(C = C)
    else:
        print("r should be  1, or 2")
        return 0

    return # the corresponding score

print(refined_classification_movies(2))
