# Utility
## Tools useful for machine learning

![tools](https://images.unsplash.com/photo-1508873535684-277a3cbcc4e8?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)

Photo by [Hunter Haley](https://unsplash.com/@hnhmarketing) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, you will find some functions, classes that I have written and saved here as Python py.file. The idea is that you can directly save and import (without the need to copy or re-write). I have documented the function, with the arguments, input, output and a example of usage. 

You can check the information directly checking the code on GitHub or if you import the function in your code, use ```help()```, for example if you import the *regression_report* and you want check the documentation type:

```Python

 help(regression_report)

```

You may write me for any request, suggestions and comments.



# Utility

| file |  Description |
|----------- | ------ |
| [Regression report](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/regression_report.py) | Print different regression metrics (similar to the classification report of scikit-learn) |
| [Upset plot](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/upset_missing_value.py) | Plot an upset plot to visualize missing data and their distribution in the columns |
| [Random NA generation](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/random_NA_generation.py) |Introduces random missing values into a dataset.|
| [Utils NA](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/utils_NA.py) | a set of utils to generate and insert NA in your dataset|
| [DR_utils](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/DR_utils.py) | a set of utils for dimensional reduction techniques|
| [Correlation_utils](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/correlation_utils.py) | a set of utils for correlation dimension|
| [Explainability_utils](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/explainability_utilies.py) | a set of utils for explainability in machine learning|

# How to import


I am providing some useful functions and classes that can be ready to use. I am providing them as executable Python files that you can import and use. You can find them in this folder.

## Google Colab

For example, if you want to use my regression_report function in Colab you can import it in this way:

```Python
import sys
import os

user = "SalvatoreRa"
repo = "tutorial"
src_dir = "machine%20learning/utility/"
pyfile = "regression_report.py" #here the name of the file py

url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{src_dir}/{pyfile}"
!wget --no-cache --backups=1 {url}
#copy here the link of the file
py_file_location = "https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/regression_report.py"
sys.path.append(os.path.abspath(py_file_location))
#here the importing
from regression_report import regression_report 
```

alternative:

```Python
!pip install wget
import wget 
wget.download('https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/machine learning/utility/utils_NA.py')

from utils_NA import *
```

## On your personal computer

If you want to import in a Jupyter Notebook or in another IDE, it is very simple. Download the file in the folder of your choice, then to import:

```Python
import sys  
sys.path.insert(0, "/path/to/the/folder/of/your/choice/")

#import the file as it was a module
import regression_report
#import function or class
from regression_report import regression_report
```

# Examples of usage

## Regression report

You can use regression report to evaluate a regression model. Regression report is returning you a report (similar to the classification report for scikit-learn) with different evaluation metrics. You can also save this regression report.

```Python
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
#boston dataset
data_dir = "https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/Boston.csv"
df = pd.read_csv(data_dir)
#separing the input features from the target variable
y = df["medv"]
X = df.drop(["medv"], axis=1)
#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 42) 
# generate an xgboost regression model
model = XGBRegressor()
#fit the model
model.fit(X_train, y_train)
#predict for the test dataset
pred = model.predict(X_test)
_ = regression_report(test_set = X_test, target = y_test, predictions = pred )
```

![regression report](https://github.com/SalvatoreRa/tutorial/blob/main/images/regression_reports.png?raw=true)

## Upset plot

You can use the upset plot to visualize missing data. Upset plot are used to visualize overlaps (you can image them as Venn diagram but they are more readable). The plot is showing the number of missing data and overlap between the different columns.

```Python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/titanic.csv")
plot_upset(data = df) 
```

![upset plot](https://github.com/SalvatoreRa/tutorial/blob/main/images/upset_plot.png?raw=true)

## Random NA generation

A simple function to insert NA randomly in your dataset.

## Utility for NA

A series of utilities for generating NA. It allows to generate MNAR, MCAR and MAR. These functions have been modified from [here]( https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values)


```Python
!pip install wget
import wget 
wget.download('https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/machine learning/utility/utils_NA.py')

from utils_NA import *
import torch
import seaborn as sns

#generate different types of NA
X_miss_mcar = produce_NA(df, p_miss=0.4, mecha="MCAR")
X_miss_mar = produce_NA(df, p_miss=0.4, mecha="MAR", p_obs=0.5)
X_miss_mnar = produce_NA(df, p_miss=0.4, mecha="MNAR", opt="logistic", p_obs=0.5)
X_miss_quant = produce_NA(df, p_miss=0.4, mecha="MNAR", opt="quantile", p_obs=0.5, q=0.3)

```

## DR_utils

I have collected here a series of functions that are not available in other packages:

**Evaluation metrics**
* Neighborhood Hit metric
* Continuity metric
* normalized stress

**Dimensional reduction plots**
* Shepard diagram

## correlation_utils

I have collected here a series of functions that are not available in other packages or I was annoyed to have to import multiple packages. I wanted something simple:

* xicor, correlation described in this [article](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1758115)
* Kendall Tau correlation
* Spearman rank correlation
* point-biserial correlation coefficient
* Goodman and Kruskal's gamma correlation
* Cramér's V statistic for a given contingency table
* mutual information-based coefficient R

## Explainability_utils

A set of utilities for explainability in machine learning. There are different functions to explain models and to plot them. Currently implemented:
* feature_importance_XGBoost - Calculate and aggregate feature importances using different methods for an XGBoost model.
* plot_feature_importance_heatmap - Plots a heatmap of the feature importance
* plot_feature_importance_barplot - Plots barplots of the feature importance DataFrame for each feature, showing mean and standard deviation. 
* plot_feature_importance_boxplot - Plots boxplots of the feature importance DataFrame for each feature.
* plot_feature_correlation_graph - Plots a graph of feature correlations and importance.


```Python

 !pip install wget 
wget.download('https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/machine learning/utility/explainability_utilies.py')

from explainability_utilies import *

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
url = 'https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/pol.csv'
df = pd.read_csv(url, sep= ';')

X = df.drop(columns=['target'])
y = df['target']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
num_round = 100
model  = xgb.XGBClassifier(objective='binary:logistic', 
                                    use_label_encoder=False, eval_metric='logloss')
model = model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)
predictions = [round(value) for value in preds]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

importance = feature_importance_XGBoost(columns_name=X.columns, _model = model, 
                           data = X_test, target=y_test )
plot_feature_importance_heatmap(importance, num_features=10)
plot_feature_importance_boxplot(importance, num_features=10)
plot_feature_importance_barplot(importance, num_features=10)
```


&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
