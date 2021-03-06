# Datasets
## Datasets for machine learning

![Library](https://images.unsplash.com/photo-1521587760476-6c12a4b040da?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80)

Photo by [Iñaki del Olmo](https://unsplash.com/@szmigieldesign) on [Unsplash](https://unsplash.com/@inakihxz)

&nbsp;

In this section, you will find a collection of datasets for machine learning project. I am curating here a selection of datasets that you can use for different tasks and I am using in my tutorials. I am adding also some notebook where I show and explain in details the dataset (making easy to use them). Check the **Google Colab** notebook, I am presenting the dataset providing information (history, context, additional information) but also showing visualization techniques for exploratory data analysis. The idea is the a ML beginner (but also who is interested in data science) can find a selection of datasets appropriate for different tasks: Moreover, the notebook can help to have a quick view of the datasets, to gain data visualization idea. In addition, I am using these datasets in my tutorial on Machine learning and artificial intelligence, provided real case uses and explain in details algorithm.

I am storing here the datasets as CSV file, if bigger than 50 MB I uploading the zip file. Check below how to import in Colab a zip file.

I will add other datasets soon. You may write me for any request, suggestions and comments.

# Datasets and Notebooks

Here are listed all the datasets in this repository, there are also the associated colab file. Check also the first notebook to quick check how to load any of this datasets.

| #| Dataset | Notebook | Source | Description |
| -| ------- | ----------- | ------ |------ |
| -| [Quick look up](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/Quick_lookup.ipynb) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/Quick_lookup.ipynb) | --- | A quick look up to how to read any of the datasets - NOTEBOOK in construction|
| 1. | [Boston house price](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/Boston.csv) | --- | [source](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) | Dataset for regression - NOTEBOOK NOT READY YET|
| 2.| [White wine dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/winequality-white.csv) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/wine_dataset.ipynb)| [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for regression/classification - NOTEBOOK NOT READY YET|
| 3.| [Red wine dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/winequality-red.csv) | --- | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for regression/classification - NOTEBOOK NOT READY YET|
| 4.| [Titanic dataset](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/titanic.csv) | --- | [source](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html) | Dataset for classification - NOTEBOOK NOT READY YET|
| 5.| [IMDB review](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip) | --- | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset for sentimental analysis, NLP tasks - NOTEBOOK NOT READY YET|
| 6.| [Word cities](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/worldcities.csv) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/word_cities_dataset.ipynb) | [source](https://archive.ics.uci.edu/ml/datasets/wine) | Dataset of the cities in the world |
| 7.| [Credit Fraud Detection](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/credit_card.csv) | [Notebook](https://github.com/SalvatoreRa/tutorial/blob/main/datasets/credit_fraud.ipynb) | [source](https://mlg.ulb.ac.be/wordpress/portfolio_page/defeatfraud-assessment-and-validation-of-deep-feature-engineering-and-learning-solutions-for-fraud-detection/) | imbalanced dataset |
| 8.| [Penguin](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/penguins.csv) | --- | [source](https://cran.r-project.org/web/packages/palmerpenguins/readme/README.html) | classification of the penguin species- NOTEBOOK NOT READY YET|
| 9.| [Mushroom](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/mushrooms.csv) | --- | [source](https://archive.ics.uci.edu/ml/datasets/mushroom) | classification of the mushroom species- NOTEBOOK NOT READY YET|
| 10.| [Iris dataset](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/iris_flowers.csv) | --- | [source](https://archive.ics.uci.edu/ml/datasets/mushroom) | classification of the mushroom species- NOTEBOOK NOT READY YET|
| 11.| [FIF21 dataset](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/FIFA_players_21.csv) | --- | [source](https://www.kaggle.com/stefanoleone992/fifa-21-complete-player-dataset?select=players_21.csv) | FIF21 player from kaggle- NOTEBOOK NOT READY YET|
| 12.| [CORA dataset: network](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/cora.cites)| --- | [source](https://linqs.soe.ucsc.edu/data) | citation network of scientific articles (two files: network and node attributes)|
|   |[CORA dataset: Node features](https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/cora.content) | --- | [source](https://linqs.soe.ucsc.edu/data) | citation network of scientific articles (two files: network and node attributes)|

&nbsp;

# Dataset suggestion

A quick chart about which dataset to use for different tasks.

&nbsp;

![dataset task](https://github.com/SalvatoreRa/tutorial/blob/main/images/datasets.png?raw=true)

&nbsp;

# Usage in Python

To use the dataset in your project you can download them (but also remember that pandas read also CSV file from the web) or if you use in colab:

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

## Usage in R

It is very easy to direct in R

```R
#example for a dataset
#you can read from directory or directly from url
df <-read.csv("https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/iris_flowers.csv")
head(df)

```
&nbsp;

# Additional Datasets 

I am listing here **additional dataset** that you can use with the provided link. The datasets listed cover different aspects of Machine learning and Artificial intelligence common tasks (computer Vision, Financial Analysis, Sentimental Analysis, Natural Language Processing, Autonomous Vehicles). If the dataset is available in multiple sources I am also adding alternative links (I check regularly and update the links that not working anymore).

| dataset |  Source | Description |
| ------- |  ------ |------ |
| US Healthcare Info | [link](https://www.kaggle.com/maheshdadhich/us-healthcare-data) | A survey of the US school system’s finances |
| Image net | [link](https://image-net.org/) | The image dataset. If you really do not know: Hundred of thousands of photo from a thousand categories  |
| LSUN | [link](https://www.tensorflow.org/datasets/catalog/lsun) | Large scale images showing different objects from given categories like bedroom, tower etc.  |
| MS COCO | [link](https://cocodataset.org/) | Segmentation, comprehension and captioning of pictures.  |
| COIL-100 | [link](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php) | 100 different 360-rotation objects  |
| Visual Genoma | [link](http://visualgenome.org/) | Visual Genome is a dataset, a knowledge base, an ongoing effort to connect structured image concepts to language. |
| Google opena dataset| [link](https://ai.googleblog.com/2016/09/introducing-open-images-dataset.html) | 9 million URLs to images that have been annotated with labels spanning over 6000 categories  |
| Stanford Dogs Dataset | [link](http://vision.stanford.edu/aditya86/ImageNetDogs/) | The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world.  |
| Indoor Scene Recognition | [link](http://web.mit.edu/torralba/www/indoor.html) | The database contains 67 Indoor categories, and a total of 15620 images. The number of images varies across categories, but there are at least 100 images per category |
| VQA  | [link](https://visualqa.org/) | VQA is a  dataset containing open-ended questions about images. More than 250000 images and questions that require an understanding of vision, language and commonsense knowledge to answer |
| Multi-Domain Sentiment Dataset | [link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) | The Multi-Domain Sentiment Dataset contains product reviews taken from Amazon.com from many product types (domains). Some domains (books and dvds) have hundreds of thousands of reviews. Others (musical instruments) have only a few hundred. A bit old but used in different several scientific articles |
| Sentiment140  | [link](http://help.sentiment140.com/for-students/) | 160000 tweet, The data is a CSV with emoticons removed and polarity annotated (negative, neutral, positive) |
| Twitter US Airline Sentiment | [link](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) | A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015  |
| Enron Email Dataset | [link](https://www.cs.cmu.edu/~./enron/) | It contains data from about 150 users, mostly senior management of Enron, organized into folders. The corpus contains a total of about 0.5M messages.  |
| Amazon reviews | [link](https://snap.stanford.edu/data/web-Amazon.html)  [link](http://jmcauley.ucsd.edu/data/amazon/) | This dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review |
| Google Books Ngrams | [link](https://aws.amazon.com/it/datasets/google-books-ngrams/) | A data set containing Google Books n-gram corpora  |
| Blog Authorship Corpus| [link](https://www.kaggle.com/rtatman/blog-authorship-corpus) | This dataset contains text from blogs written on or before 2004. Posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus incorporates a total of 681,288 posts and over 140 million words |
| Wikipedia Links Data | [link](https://code.google.com/archive/p/wiki-links/downloads) [link](http://www.iesl.cs.umass.edu/data/data-wiki-links)|  the Wikilinks dataset comprising of 40 million mentions over 3 million entities  |
| Hansards Text Chunks from the Canadian Parliament |  [link](https://metatext.io/datasets/hansards-canadian-parliament)|  Created by Natural Language Group - USC at 2001,  Dataset contains pairs of aligned text chunks (sentences or smaller fragments) from the official recordsof the 36th Canadian Parliament. in English language. Containing 1.3M in Text file format  |
| Jeopardy |  [link](https://www.kaggle.com/tunguz/200000-jeopardy-questions)|  over 200,000 questions from the Jeopardy quiz show |
| SMS Spam Compilation in English |  [link](https://www.kaggle.com/uciml/sms-spam-collection-dataset)|  The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages |
| SMS Spam Compilation in English |  [link](https://www.yelp.com/dataset)|  More than 8 million reviews, 160000 bussiness, are included in an open dataset  by Yelp |
| Berkeley DeepDrive BDD100k |  [link](https://bdd-data.berkeley.edu/)| More than 100,000 views of driving journeys of over 1,100 hours through various periods of the day and weather conditions |
| Baidu Apolloscapes |  [link](http://apolloscape.auto/)| Trajectory dataset, 3D Perception Lidar Object Detection and Tracking dataset including about 100K image frames, 80k lidar point cloud and 1000km trajectories for urban traffic. The dataset consisting of varying conditions and traffic densities which includes many challenging scenarios where vehicles, bicycles, and pedestrians move among one another. |
| comma2k19 |  [link](https://github.com/commaai/comma2k19)|  a dataset of over 33 hours of commute in California's 280 highway |
| Oxford’s Robotic Car |  [link](https://robotcar-dataset.robots.ox.ac.uk/)| TThe Oxford RobotCar Dataset contains over 100 repetitions of a consistent route through Oxford, UK, captured over a period of over a year. The dataset captures many different combinations of weather, traffic and pedestrians, along with longer term changes such as construction and roadworks. |
| Cityscape Dataset |  [link](https://www.cityscapes-dataset.com/)| stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames |
| Traffic Sign Recognition  |  [link](http://apolloscape.auto/)| More than 10000+ traffic sign annotations |
| LISA traffic light |  [link](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset)| The database consists of continuous test and training video sequences, totaling 43,007 frames and 113,888 annotated traffic lights. |
| National Institute of Health X-Ray Dataset| [link](https://medpix.nlm.nih.gov/home) | This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning.  |


&nbsp;

# Suggested Repositories 

I am listing here additional repositories that you can use with the provided link and a description.

The list is not exhaustive and I am planning to extend, please feel free to suggest addition.

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
| Project Gutenberg| [link](https://www.gutenberg.org/) | Project Gutenberg is a library of over 60,000 free eBooks |
| LINQS| [link](https://linqs.soe.ucsc.edu/data) | different relational dataset for graph analysis |
| PyTorch Geometric| [link](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) | many different relational dataset for graph analysis |

&nbsp;

# Contributing

Feel free to suggest other datasets and repositories


# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
