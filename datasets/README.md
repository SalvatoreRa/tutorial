# Datasets
## Datasets for machine learning

![tree classifiers](https://github.com/SalvatoreRa/tutorial/blob/main/images/lukasz-szmigiel-jFCViYFYcus-unsplash.jpg?raw=true)

Photo by [Lukasz Szmigiel](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, you will find a collection of datasets for machine learning project. I am curating here a selection of datasets that you can use for different tasks and I am using in my tutorials. I am adding also some notebook where I show and explain in details the dataset (making easy to use them).

I am storing here the datasets as CSV file, if bigger than 50 MB I uploading the zip file. Check below how to import in Colab a zip file.

You may write me for any request, suggestions and comments.

# Tutorial

| dataset | Notebook | Description |
| ------- | ----------- | ------ |
| [Data manipulation](https://) | --- | Common data manipulation tasks and data issues - NOTEBOOK NOT READY YET|


&nbsp;

# Usage

To use the dataset in your project you can download them or if you use in colab:

for CSV file 

```Python
#example for a dataset
#you can read from directory or directly from url
data_dir = "https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/Boston.csv"
df = pd.read_csv(data_dir)
```

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



&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
