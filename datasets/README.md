# Datasets
## Datasets for machine learning

![tree classifiers](https://github.com/SalvatoreRa/tutorial/blob/main/images/lukasz-szmigiel-jFCViYFYcus-unsplash.jpg?raw=true)

Photo by [Lukasz Szmigiel](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, you will find a collection of datasets for machine learning project. I am curating here a selection of datasets that you can use for different tasks and I am using in my tutorials. I am adding also some notebook where I show and explain in details the dataset (making easy to use them)

You may write me for any request, suggestions and comments.

# Tutorial

| Tutorial | Notebook | Description |
| ------- | ----------- | ------ |
| [Data manipulation](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/data_manipulation.ipynb) | Common data manipulation tasks and data issues - MEDIUM ARTICLE NOT YET PUBLISHED|


&nbsp;

# Usage

To use the dataset in your project you can download them 

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

| file |  Description |
|----------- | ------ |
| [Regression report](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/regression_report.py) | Print different regression metric (similar to classification report of scikit-learn) |

&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
