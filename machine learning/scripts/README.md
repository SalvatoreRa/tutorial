# Scripts
## Python and R scripts that you can execute

![tree classifiers](https://github.com/SalvatoreRa/tutorial/blob/main/images/lukasz-szmigiel-jFCViYFYcus-unsplash.jpg?raw=true)

Photo by [Lukasz Szmigiel](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/)

&nbsp;

This section is dedicated 


&nbsp;

# Utility

I am providing some useful fuctions and classes that can be ready to use. I am providing them as executable python file that you can import and use. You find them in the **utility folder**.

Check in the utiliy folder the example of usages and the explanation about them. Each function is document and you can access the provided documentation.

For example, if you want to use my regression_report function in Colab you can import in this way:

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

Or alternatively, you can use in this way in Colab:

```Python
wget.download('https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/machine learning/utility/utils_NA.py')
!pip install wget 
from utils import *
import torch
import seaborn as sns

#generate different type of NA
X_miss_mcar = produce_NA(df, p_miss=0.4, mecha="MCAR")
X_miss_mar = produce_NA(df, p_miss=0.4, mecha="MAR", p_obs=0.5)
X_miss_mnar = produce_NA(df, p_miss=0.4, mecha="MNAR", opt="logistic", p_obs=0.5)
X_miss_quant = produce_NA(df, p_miss=0.4, mecha="MNAR", opt="quantile", p_obs=0.5, q=0.3)

```


| File |  Description |
|----------- | ------ |
| [Regression report](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/regression_report.py) | Print different regression metric (similar to classification report of scikit-learn) |
| [Upset plot](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/upset_missing_value.py) | Plot an upset plot to visualize missing data and their distribution in the columns |
| [Random NA generation](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/random_NA_generation.py) |Introduces random missing values into a dataset.|
| [Utils NA](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/utils_NA.py) | a set of utils to generate and insert NA in your dataset|


&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
