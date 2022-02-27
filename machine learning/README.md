# Tutorial
## Python and R tutorial on Machine Learning

![tree classifiers](https://github.com/SalvatoreRa/tutorial/blob/main/images/lukasz-szmigiel-jFCViYFYcus-unsplash.jpg?raw=true)

Photo by [Lukasz Szmigiel](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, you will find the **Jupiter Notebook** for the the tutorial I published in **Medium**. I suggest to read the tutorial and the companion tutorial code in the order provided in the table below. For practical reason, I have divided some of the tutorial in more than one part (allowing to concentrate in one of the tutorial on the theoretical part and in the others about the programming). Tutorial dedicated only to the theory have not a linked Jupiter notebook containing the **Python** code used for the model and the graph. I wrote and test the code in Google Colab in order to make it reproducible.

I am progressively adding also some **R tutorials**, I decided to upload the R-scripts so you can tested them. Check the table below where I list the Colab Notebooks, the R-scripts and the companion articles.

Moreover, you may find here some colab notebook without a theoretical tutorial (yet). I decided to upload the code before I have finish to write the theoretical part (this would be indicated). I am convinced that the code alone is already beneficial. I would successively publish on Medium the written article (with details and comment to the code).

You may write me for any request, suggestions and comments.

# Tutorial

| Tutorial | Notebook | Description |
| ------- | ----------- | ------ |
| [Data manipulation](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/data_manipulation.ipynb) | Common data manipulation tasks and data issues - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Pandas Cheatsheet](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Pandas_summary.ipynb) | Introduction to Pandas library - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Regular expression in Python](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Regular_expression.ipynb) | regular expression in Python - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Matrix operations for machine learning](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/matrix_operations.ipynb) | Matrix operations for machine learning in Python - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Matrix operations for machine learning - part 2](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/matrix_operations_part2.ipynb) | Matrix operations for machine learning in Python, the second part - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Tree classifiers](https://) | ---- | Introduction to tree classifiers, theory and math explained simple - MEDIUM ARTICLE NOT YET PUBLISHED |
| [Tree classifiers](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/training_tree_classifier.ipynb) | Training of tree classifiers - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Visualize decision tree](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Visualize_decision_tree.ipynb) | Visualization of decision tree - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Train and visualize decision tree in R](https://) | [R-script](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/decision_tree_in_R.R) | Plot and visualize a decision tree in R - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Evaluation metrics for classification - part I](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/classc_metr.ipynb) | How to calculate, code, and interpret evaluation metrics for classification - MEDIUM ARTICLE NOT YET PUBLISHED |
| [Evaluation metrics for classification - part II](https://) | --- | Part II about imbalance dataset and multiclass classification - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Linear Regression - OLS](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Least_squares_regression.ipynb) | Linear regression introduction, least square method - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Evaluation metrics for regression](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/regression_metrics.ipynb)  | Evaluation metrics for regression - MEDIUM ARTICLE NOT YET PUBLISHED|
| [Train and visualize regression tree](https://) | [notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Train_and_Visualize_regression_tree.ipynb)  | Train, visualize regression decision tree in Python- MEDIUM ARTICLE NOT YET PUBLISHED|
| [Linear regression in R](https://) | [R-script](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/linear%20regression%20in%20r.R)  | Train and visualize a linear regression model in R- MEDIUM ARTICLE NOT YET PUBLISHED|
| Introduction to NetworkX | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/Intro_to_Networkx.ipynb)  | A notebook to refresh the use of NetworkX|

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

| file |  Description |
|----------- | ------ |
| [Regression report](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/regression_report.py) | Print different regression metric (similar to classification report of scikit-learn) |
| [Upset plot](https://github.com/SalvatoreRa/tutorial/blob/main/machine%20learning/utility/upset_missing_value.py) | Plot an upset plot to visualize missing data and their distribution in the columns |

&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
