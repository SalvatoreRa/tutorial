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

I am providing some useful fuctions and classes that can be ready to use. I am providing them as executable python file that you can import and use. You find them in this folder.

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

&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github