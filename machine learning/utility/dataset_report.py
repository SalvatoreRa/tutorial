#!/usr/bin/env python3

# the function here are intended to conduct quick exploration of the dataset
# currently implemented: missing value
# this will expanded with time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def missing_values_table(df):
        # credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    
def classification_EDA(X_df = None,target= None):
    '''
    Provide a description of the dataset for classification
    missing value in X and target variable
    number of examples for each classes and the relative percentage to identify if the dataset is unbalanced
    
    '''
    missing_table= missing_values_table(X_df)
    print('Dataset')
    print(missing_table)
    print('Target variable')
    target = pd.DataFrame(target, columns= ['target'])
    missing_target= missing_values_table(target)
    print(missing_target)
    target_count = target.target.value_counts()
    print('Target classes: ' + str(target_count.index.to_list()))
    print('number examples for class: ' + str(target_count.values))
    print('percentage for class: ' + str(target_count.values/np.sum(target_count.values)*100))