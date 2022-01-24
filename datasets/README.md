# Datasets
## Datasets for machine learning

![Library](https://images.unsplash.com/photo-1521587760476-6c12a4b040da?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)

Photo by [Iñaki del Olmo](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/@inakihxz)

&nbsp;

In this section, you will find a collection of datasets for machine learning project. I am curating here a selection of datasets that you can use for different tasks and I am using in my tutorials. I am adding also some notebook where I show and explain in details the dataset (making easy to use them). Check the **Google Colab** notebook, I am presenting the dataset providing information (history, context, additional information) but also showing visualization techniques for exploratory data analysis. The idea is the a ML beginner (but also who is interested in data science) can find a selection of datasets appropriate for different tasks: Moreover, the notebook can help to have a quick view of the datasets, to gain data visualization idea. In addition, I am using these datasets in my tutorial on Machine learning and artificial intelligence, provided real case uses and explain in details algorithm.

I am storing here the datasets as CSV file, if bigger than 50 MB I uploading the zip file. Check below how to import in Colab a zip file.

I will add other datasets soon. You may write me for any request, suggestions and comments.

# Tutorial

| Dataset | Notebook | Source | Description |
| ------- | ----------- | ------ |------ |
| [Boston house price](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/Boston.csv) | --- | [source](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) | Dataset for regression - NOTEBOOK NOT READY YET|
| [White wine dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/winequality-white.csv) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/wine_dataset.ipynb)| [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for regression/classification - NOTEBOOK NOT READY YET|
| [Red wine dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/winequality-red.csv) | --- | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for regression/classification - NOTEBOOK NOT READY YET|
| [Titanic dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/titanic.csv) | --- | [source](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html) | Dataset for classification - NOTEBOOK NOT READY YET|
| [IMDB review](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip) | --- | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for sentimental analysis, NLP tasks - NOTEBOOK NOT READY YET|
| [Word cities](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/worldcities.csv) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/word_cities_dataset.ipynb) | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset of the cities in the world |
| [Credit Fraud Detection](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/credit_card.csv) | --- | [source](https://mlg.ulb.ac.be/wordpress/portfolio_page/defeatfraud-assessment-and-validation-of-deep-feature-engineering-and-learning-solutions-for-fraud-detection/) | imbalanced dataset - NOTEBOOK NOT READY YET|


&nbsp;

# Dataset suggestion

A quick chart about which dataset to use for different tasks.

&nbsp;

![dataset task](https://github.com/SalvatoreRa/tutorial/blob/main/images/datasets.png?raw=true)

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

If the file is contained in a zip.file, to upload in Google Colab


```Python
import sys
import os
#this for unzip and read the file
!wget https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip?raw=true
!unzip IMDB.zip?raw=true
imdb_data=pd.read_csv("IMDB Dataset.csv")
```
&nbsp;

# Additional Datasets 

I am listing here additional dataset that you can use with the provided link

| dataset |  Source | Description |
| ------- |  ------ |------ |
| US Healthcare Info | [link](https://www.kaggle.com/maheshdadhich/us-healthcare-data) | A survey of the US school system’s finances |

&nbsp;

# Suggested Repositories 

I am listing here additional dataset that you can use with the provided link

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| Kaggle | [link](https://www.kaggle.com/) | A great source for datasets, with code and competitions. |
| UCI machine learning | [link](https://archive.ics.uci.edu/ml/index.php) | One of the oldest repository, user contributed. Not all the dataset are clean, but you can download without  |
| School System Finance | [link](https://archive.ics.uci.edu/ml/index.php) | One of the oldest repository, user contributed. Not all the dataset are clean, but you can download without  |
| EU Open Data Portal | [link](https://data.europa.eu/en) | More than a 1 million datasets released by the EU about health, finance, science etc... |
| Data. gov | [link](https://www.data.gov/) | Data from various US government departments, but it can tricky to exploit the data |
| Quandl | [link](https://www.data.gov/) | Great source of economic and financial data |
| Word Bank | [link](https://data.worldbank.org/) | You can download data about population demographics, global economic and development indicators |
| IMF statistics | [link](https://www.imf.org/en/Data) | International Monetary Fund data about international finaces, debt rate, etc... |
| American Economic Association  | [link](https://data.worldbank.org/) | US macroeconomic data concerning employment, economic output, and other  variables |
| Google Trends| [link](https://trends.google.com/trends/?q=google&ctab=0&geo=all&date=all&sort=0) | Internet search activity details and  trends |

&nbsp;

# Contributing



# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
