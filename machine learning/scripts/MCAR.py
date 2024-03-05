#!/usr/bin/env python3

print('starting importing')

# Parameters

datasets_name = ['pol', 'house_16H', 'MagicTelescope']

ms = 'MCAR'
perc = 0.4
fold_splits = 5

# if status fast, MIDAS will be not executed otherwise, since the slower will be not executed

fast_track = True

print('fast track status:')
print(fast_track)
if fast_track:
    print('when fast track active, MIDAS is not executed')

########### Description ################ 
# This script is for a small benchmarking of missing data
# It is using XGBoost as a model and different metrics to test effect of missing values
# All the algorithms are described in the correspective article, check it
# Correspective tutorial and others: https://github.com/SalvatoreRa/tutorial
# Feel free to use, in case you find useful for your research cite it 



import pandas as pd
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import average_precision_score, matthews_corrcoef, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# Use wget download only once, you can just download from github and leave the utils_NA.py in the folder


#import wget
#wget.download('https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/machine learning/utility/utils_NA.py')

from utils_NA import *
import torch
import seaborn as sns
import time
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from datasets import load_dataset
from qolmat.imputations.diffusions.ddpms import TabDDPM
from qolmat.benchmark import metrics
from qolmat.imputations import imputers

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import MIDASpy as md

print('import complete')



start_time = time.time()


measures = ['dataset', 'missing value', 'percentage', 'imputation', 'score', 'MAE', 'RMSR',
           'Kullback-Leibler', 'Energy distance', 'Fréchet', 'Wasserstein', 'time']

results = pd.DataFrame( columns = measures)

for j in datasets_name:
    #select the dataset
    # you can find additional dataset in the repository
    # here we are testing only three datasets but you can enlarge the comparison
    data_dir = "https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/"
    
    df = pd.read_csv(data_dir+j +'.csv',sep=';')
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    lb = LabelBinarizer()
    y =lb.fit_transform(y)
    
    # first we just start with the baseline
    # here we use as a classifier XGBoost but you can change with another on
    # we do 5 fold cross validation
    kf = KFold(n_splits=fold_splits)
    X = np.array(X)
    for train_index, test_index in kf.split(X):
        algo_time = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = xgb.XGBClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
               

        m = [j, 'none', 'none', 'none', accuracy_score(y_test, y_test_pred), 
             0, 0,0,0,0,0, time.time() - algo_time]
        results.loc[len(results)] = m
        
    # we are introducing missing data in different percentage
    # feel free to change the percentage that you prefer
    # here we are using just 4 for reducing time
    
    print('missing data introduction')
    for perc in [0.1, 0.3, 0.4, 0.5]:
        
        print("The missing value percentage is {}".format(perc))
        
        X_miss_mcar = produce_NA(df.iloc[:, :-1], p_miss=perc, mecha="MCAR")
        X_miss_mcar = X_miss_mcar['X_incomp'].detach().numpy()
        X_miss_mcar = np.where(X_miss_mcar=='nan', np.nan, X_miss_mcar )

        kf = KFold(n_splits=fold_splits)
        

        for train_index, test_index in kf.split(X_miss_mcar):
            
            
            
            X_train, X_test = X_miss_mcar[train_index], X_miss_mcar[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # we are starting now the process of imputation
            # for each case we will measure the score
            # the score represents the accuracy in this case
            # we are also measure the total time needed from imputation to classification
            # we are also measuring the accuracy of the reconstruction
            # for example MAE is the absolute error when you compare the imputed and the 
            # original dataset
            # Qolmat allows different measure so we will use different ones and store them
            # for more information about these metrics: https://qolmat.readthedocs.io/en/latest/api.html#metrics
            # check also here: https://dcor.readthedocs.io/en/latest/theory.html
            print('Start imputation')


            #### Imputing with mean
            # we are using here just the mean, basically for each feature we use the mean value
            # we are using scikit-learn simple imputer which is fast and used implementation
            # more information about imputation with scikit-learn here: https://scikit-learn.org/stable/modules/impute.html
            # additional information about simpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
            # Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
            
            print('Imputation with Mean')
            algo_time = time.time()
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X_train)
            X_train_imp =imp.transform(X_train)
            X_test_imp =imp.transform(X_test)

            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            # we are using different metrics to check here the difference between imputed
            # and real dataset
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'Mean', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with the Median
            # we are using here just the median, the principle is the same for the mean
            # we are using the median because is common method
            print('Imputation with Median')
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
            imp.fit(X_train)
            X_train_imp =imp.transform(X_train)
            X_test_imp =imp.transform(X_test)

            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)

            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'median', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with 0
            # we are zero-imputation or called constant imputation
            # you can use another value but it is common practice using the zero
            print('Imputation with zero')
            algo_time = time.time()
            imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
            imp.fit(X_train)
            X_train_imp =imp.transform(X_train)
            X_test_imp =imp.transform(X_test)

            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'zero', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with KNN imputer
            # we are testing two different option for the KNNimputer one from scikit-learn
            # this is the scikit-learn version
            # Imputation for completing missing values using k-Nearest Neighbors.
            # notice that the impiuter use the values from the neighbours
            # additional information here: https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
            print('Imputation with KNN - scikit')
            algo_time = time.time()
            imp = KNNImputer(missing_values=np.nan)
            imp.fit(X_train)
            X_train_imp =imp.transform(X_train)
            X_test_imp =imp.transform(X_test)

            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'KNN scikit', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with Iterative imputation
            # The IterativeImputer class is very flexible - it can be used with a variety of estimators to do round-robin regression, treating every variable as an output in turn.
            # Multivariate imputer that estimates each feature from all the others.

            #A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.
            # This come from scikit-learn and can be seen as MICE
            # for more info: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
            # and also here: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.htm
            #
            print('Imputation with Iterative imputer')
            algo_time = time.time()
            imp = IterativeImputer(random_state=0)
            imp.fit(X_train)
            X_train_imp =imp.transform(X_train)
            X_test_imp =imp.transform(X_test)
            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'Iterative', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with TabDDPM
            # Diffusion model for tabular data based on Denoising Diffusion Probabilistic Models (DDPM)
            # diffusion model, from Qolmat
            # here for more infor: https://qolmat.readthedocs.io/en/latest/generated/qolmat.imputations.diffusions.ddpms.TabDDPM.html
            # original article: https://arxiv.org/abs/2107.03502
            #
            # 
            print('Imputation with TabDDPM')
            algo_time = time.time()
            imp = TabDDPM()

            #X_train_imp =imp.fit_transform(pd.DataFrame(X_train[:100,:]))
            imp.fit(pd.DataFrame(X_train))
            X_train_imp =imp.predict(pd.DataFrame(X_train[:,:]))
            imp.fit(pd.DataFrame(X_test))
            X_test_imp =imp.predict(pd.DataFrame(X_test[:,:]))
            #X_test_imp =imp.fit_transform(pd.DataFrame(X_test[:100,:]))

            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'TabDDPM', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with kNN imputer - fancyimputer
            # alternative algorithm
            # 
            print('Imputation with kNN - fancy imputer')
            algo_time = time.time()
            imp = imputers.ImputerKNN()
            X_train_imp =imp.fit_transform(pd.DataFrame(X_train[:,:]))
            X_test_imp =imp.fit_transform(pd.DataFrame(X_test[:,:]))
            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train_imp, y_train)
            y_test_pred = clf.predict(X_test_imp)
            mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
            df_mask = np.array(np.full((X_train_imp.shape[0], 
                                        X_train_imp.shape[1],), True, dtype=bool))
            RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            RMSR= np.sum(RMSR)
            print(RMSR)
            KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            KL= np.sum(KL)
            print(KL)
            E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            E= np.sum(E)
            print(E)
            FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            FR= np.sum(FR)
            print(FR)
            WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                           pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
            WA= np.sum(WA)
            print(WA)

            m = [j, ms, perc, 'KNN 2', accuracy_score(y_test, y_test_pred), mae, 
                RMSR, KL, E, FR, WA, time.time() - algo_time]
            results.loc[len(results)] = m

            #### Imputing with MIDAS
            # MIDASpy is a Python package for multiply imputing missing data using deep learning methods.
            # according to the authors:  The MIDASpy algorithm offers significant accuracy and efficiency advantages over other multiple imputation strategies, particularly when applied to large datasets with complex features.
            # here for additional info: https://github.com/MIDASverse/MIDASpy/tree/master
            # 
            if not fast_track:  
                print('Imputation with MIDAS')
                algo_time = time.time()
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns =[str(i) for i in range(X_train.shape[1])] )
                na_loc = X_train_scaled.isnull()
                X_train_scaled[na_loc] = np.nan
                imputer = md.Midas(layer_structure= [16,16], vae_layer = False, seed= 89, input_drop = 0.75)
                imputer.build_model(X_train_scaled)
                imputer.train_model(training_epochs = 15)
                imputations = imputer.generate_samples(m=10).output_list
                stacked_array = np.stack(imputations)
                X_train_imp = np.mean(stacked_array, axis=0)

                scaler = MinMaxScaler()
                X_test_scaled = scaler.fit_transform(X_test)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns =[str(i) for i in range(X_test.shape[1])] )
                na_loc = X_test_scaled.isnull()
                X_test_scaled[na_loc] = np.nan
                imputer = md.Midas(layer_structure= [16,16], vae_layer = False, seed= 89, input_drop = 0.75)
                imputer.build_model(X_test_scaled)
                imputer.train_model(training_epochs = 15)
                imputations = imputer.generate_samples(m=10).output_list
                stacked_array = np.stack(imputations)
                X_test_imp = np.mean(stacked_array, axis=0)

                clf = xgb.XGBClassifier(random_state=42)
                clf.fit(X_train_imp, y_train)
                y_test_pred = clf.predict(X_test_imp)
                mae = mean_absolute_error(np.array(df.iloc[:, :-1])[train_index], X_train_imp)
                df_mask = np.array(np.full((X_train_imp.shape[0], 
                                            X_train_imp.shape[1],), True, dtype=bool))
                RMSR = metrics.root_mean_squared_error(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                               pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
                RMSR= np.sum(RMSR)
                print(RMSR)
                KL = metrics.kl_divergence(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                               pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
                KL= np.sum(KL)
                print(KL)
                E = metrics.sum_energy_distances(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                               pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
                E= np.sum(E)
                print(E)
                FR = metrics.frechet_distance(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                               pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
                FR= np.sum(FR)
                print(FR)
                WA = metrics.dist_wasserstein(pd.DataFrame(np.array(df.iloc[:, :-1])[train_index]), 
                               pd.DataFrame(X_train_imp), pd.DataFrame(df_mask))
                WA= np.sum(WA)
                print(WA)

                m = [j, ms, perc, 'MIDAS', accuracy_score(y_test, y_test_pred), mae, 
                    RMSR, KL, E, FR, WA, time.time() - algo_time]
                results.loc[len(results)] = m

            #### Training without imputing
            # different algorithm are not sensible to NA
            # for example decision tree are not sensible to the NA (they can treat them internally)
            # since ensemble are decision tree, we can try the algorithm without imputation
            # 
            # 
            print('algorithm not sensible to NA')
            algo_time = time.time()
            clf = xgb.XGBClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            m = [j, ms, perc, 'XGBoost internal', accuracy_score(y_test, y_test_pred), 
                 0, 0,0,0,0,0, time.time() - algo_time]
            results.loc[len(results)] = m

            
print('finished loop')
print('saving data')
# saving data
results.to_csv('missing_data_' + ms +'.csv')

print('plotting global')
# Plotting
# we are saving the global data in this case. 
# we are dividing for percentage for better understand the trend
# we are saving a plot for each measure
for i in ['score', 'MAE', 'RMSR','Kullback-Leibler', 'Energy distance', 
          'Fréchet', 'Wasserstein', 'time']:
    plot=sns.boxplot(data=results, x="imputation", y=i, hue ='percentage')
    plt.xticks(rotation=45)
      
    fig = plot.get_figure()
    fig.savefig(ms+ '_'+i +'.jpeg')
    plt.close()

print('plot local')
# Plotting
# we are saving the local data in this case. 
# we are dividing for percentage for better understand the trend,
# a single plot for percentage
# we are saving a plot for each measure
for j in [0.1, 0.3, 0.4, 0.5]:
    res = results[results['percentage']==j]
for i in ['score', 'MAE', 'RMSR','Kullback-Leibler', 'Energy distance', 
          'Fréchet', 'Wasserstein', 'time']:
    plot=sns.boxplot(data=res, x="imputation", y=i)
    plt.xticks(rotation=45)
      
    fig = plot.get_figure()
    fig.savefig(ms+ '_'+str(j)+'_'+i +'.jpeg')
    plt.close()

print('script finished')
print("--- %s seconds ---" % (time.time() - start_time))