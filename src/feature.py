import os
import numpy as np 
import scipy.stats as stats
import pandas as pd 

def read_data(DataPath = '.'):
    """Function to return the raw DataFrame
    ======================================
    ARGUMENTS:
    DATAPATH: '.' string 
    ======================================
    RETURN: Pandas Dataframe
    ======================================"""
    if os.path.exists(DataPath):
        return pd.read_csv(DataPath)
    else:
        print("File not Exist")
def summary(x = None):
    return x.describe().transpose().to_csv('/home/miki/Desktop/Deployment/survival-analysis/reports/summary.csv', index = False)


def outlier_remove(x = None):
    """"""
    Q1 = x.quantile(q=.25)
    Q3 = x.quantile(q=.75)
    IQR = x.apply(stats.iqr)
    #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
    cd = x[~((x < (Q1-1.5*IQR)) | (x > (Q3+1.5*IQR))).any(axis=1)]
    #find how many rows are left in the dataframe 
    return cd
def data_split(df, train_size = 0.8, test_size= 0.2):
    """_summary_

    Args:
        data (_type_): _description_
        train_size (_type_): _description_
        test_size (_type_): _description_
        stratify (_type_): _description_
    """
    from sklearn.model_selection import train_test_split
     
    X = df.drop(['Event'], axis = 1)
    y = df['Event']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= train_size, test_size = test_size,
                                                        stratify = y, random_state = 42)
    dtrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis = 1, join = 'inner')
    dtest = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis = 1, join = 'inner')
    train = dtrain.to_csv('/home/miki/Desktop/Deployment/survival-analysis/data/trainds/train.csv', index = False)
    test = dtest.to_csv('/home/miki/Desktop/Deployment/survival-analysis/data/testds/test.csv', index = False)
    return train, test