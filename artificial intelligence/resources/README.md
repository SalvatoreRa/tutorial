# Additional Resources
## Resources on the artfificial intelligence

![Resources](https://images.unsplash.com/photo-1548048026-5a1a941d93d3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80)

Photo by [Daniel](https://unsplash.com/@setbydaniel) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, I will suggest and add many resources on artificial intelligence that can be useful. The list is not exhaustive and I will expand with time, if you want to suggest other resources, you are welcome.

I am listing seminal articles (the list is clearly not exhaustive), free available books, free courses and useful tools that I found. I am also listing a large set of database where you can find dataset for different machine learning tasks. These categories are also organized with sub-categories for clarity.

&nbsp;

# **Table of Contents**

* Scientific articles
* Books
* Tools
* Dataset database
* Free courses


&nbsp;

# Scientific articles

This is a list of seminal articles on different fields of artificial intelligence. This is list is not meant to be complete (but I will expand with time and you can suggest additions). I see a starting point of articles that should be read it, I consulted most of them for my tutorials and many them present the basis of the artificial intelligence. I try to cover many field of the actual artificial intelligence research. I am listing free accessible articles that can be accessed by anyone.

## General introduction

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [How to read a paper](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)| General introduction | 1998 | A very brief introduction on how to read a scientific paper |
| [History of Deep learning](https://arxiv.org/abs/1702.07800)| General introduction | 2017 | On the Origin of Deep Learning |
| [History of Deep learning](https://arxiv.org/abs/1701.05549)| General introduction | 2017 | Deep Neural Networks - A Brief History |

## General Interest

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [ADAM](https://jmlr.org/papers/v15/srivastava14a.html)| General interest | 2014 |first paper presenting ADAM |
| [Group normalization](https://arxiv.org/pdf/1803.08494.pdf)| General interest | 2018 | discussing group normalization |
| [Dropout](https://arxiv.org/abs/1412.6980)| General interest | 2014 |first paper on dropout |
| [System design](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)| General interest | 2015 | Hidden Technical Debt in Machine Learning Systems |

## Computer vision

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [LeNet-5](http://yann.lecun.com/exdb/publis/index.html#lecun-98)| computer vision | 1998 |showing that you can stack convolutional layers instead of dense layers |
| [AlexNet ](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)| computer vision | 2012 |first showing the use of ReLu |
| [VGG-16 ](https://arxiv.org/abs/1409.1556)| computer vision | 2014 |first very deep network |
| [Inception-v1 ](https://arxiv.org/abs/1409.4842)| computer vision | 2014 |first models stacking block and not layers (inside a block can be different layers) |
| [Inception-v3 ](https://arxiv.org/abs/1409.1556)| computer vision | 2015 |Among the first designers to use batch normalisation |
| [Res-Net50 ](https://arxiv.org/abs/1512.03385)| computer vision | 2015 |made popular skip connections (resnet block) |
| [Xception](https://arxiv.org/abs/1610.02357)| computer vision | 2016 |depthwise separable convolution layers |
| [Inception-v4 ](https://arxiv.org/abs/1602.07261)| computer vision | 2016 |inception block and residual connection |
| [Inception-ResNet-V2 ](https://arxiv.org/abs/1602.07261)| computer vision | 2016 |Residual Inception blocks |
| [Batch normalization](https://arxiv.org/abs/1502.03167)| computer vision | 2015 |Batch normalization |
| [self attention and convolution](https://arxiv.org/abs/2111.14556)| computer vision | 2021 |On the Integration of Self-Attention and Convolution|
| [ConvNet versus Transformers](https://arxiv.org/pdf/2201.03545.pdf)| computer vision | 2021 |A ConvNet for the 2020s: interesting paper on the future of convnet|

## Natural language processing (NLP)

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [Word2vec](https://arxiv.org/pdf/1301.3781v3.pdf)| NLP | 2013 | Word2vec, word embedding |
| [Layer normalization](https://arxiv.org/abs/1607.06450)| NLP | 2013 | layer normalization |
| [RNN ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2015 |Visualizing and Understanding Recurrent Networks |
| [LSTM ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2015 |Review about LSTM |
| [GRU ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2014 |Gated recurrent unit |
| [Transformers ](https://arxiv.org/abs/1706.03762)| NLP | 2016 |First paper proposing transformers |
| [Question Answering ](https://arxiv.org/abs/1704.00051)| NLP | 2017 |Reading Wikipedia to Answer Open-Domain Questions |
| [BERT ](https://arxiv.org/abs/1810.04805)| NLP | 2018 | First paper proposing BERT |
| [DistilBERT ](https://arxiv.org/abs/1910.01108)| NLP | 2019 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter |
| [SciBERT ](https://arxiv.org/abs/1903.10676)| NLP | 2019 | SciBERT: A Pretrained Language Model for Scientific Text |
| [BART ](https://arxiv.org/abs/1910.13461)| NLP | 2019 | BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension |
| [ Text Summarization ](https://arxiv.org/abs/2010.04529)| NLP | 2020 | What Have We Achieved on Text Summarization |
| [Transformers ](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)| NLP | 2018 |GPT-1 paper |
| [Transformers ](https://openai.com/blog/better-language-models/)| NLP | 2018 |GPT-2 paper |
| [Transformers ](https://arxiv.org/abs/2005.14165)| NLP | 2020 |GPT-3 paper |
| [Attention reivew ](https://arxiv.org/abs/1904.02874)| NLP | 2019 | An Attentive Survey of Attention Models |
| [Transformers ](https://arxiv.org/abs/2105.14103)| NLP | 2021 | An Attention Free Transformer |

## Generative Learning

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [GAN ](https://arxiv.org/pdf/1406.2661v1.pdf)| Generative Learning | 2014 |Generative Adversarial Nets |
| [GAN ](https://arxiv.org/abs/1511.06434)| Generative Learning | 2015 |original article presenting DC-GAN |


## Tabular data

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [Tabular data](https://arxiv.org/abs/2012.06678)| Tabular data | 2020 | TabTransformer: Tabular Data Modeling Using Contextual Embeddings |
| [Tabular data](https://arxiv.org/abs/2106.03253)| Tabular data | 2021 |Tabular Data: Deep Learning is Not All You Need (showing that deep learning with tabular data is still not the first choice) |
| [Tabular data](https://arxiv.org/pdf/2106.01342.pdf)| Tabular data | 2021 |Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training |
| [Tabular data](https://arxiv.org/pdf/2110.01889.pdf)| Tabular data | 2021 | Method's review on Deep Neural Networks and Tabular Data |
| [Tabular data](https://arxiv.org/pdf/2106.11189.pdf)| Tabular data | 2021 | how regularization impact Neural Network for tabular data|

## Reinforcement learning

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [RL](https://arxiv.org/abs/1312.5602)| Reinforcement Learning | 2013 | Playing Atari with Deep Reinforcement Learning |
| [RL](https://arxiv.org/abs/2201.03916)| Reinforcement Learning | 2022 | Automated Reinforcement Learning (AutoRL): A Survey and Open Problems |

## Geometric deep learning

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [GNN](https://arxiv.org/abs/1611.08097)| Geometric learning | 2016 | Geometric deep learning: going beyond Euclidean data |
| [GNN](https://arxiv.org/abs/1609.02907)| Geometric learning | 2016 | Semi-Supervised Classification with Graph Convolutional Networks |
| [Knowledge graph review](https://iopscience.iop.org/article/10.1088/1742-6596/1487/1/012016)| Geometric learning | 2020 | A Survey on Application of Knowledge Graph |
| [Graph transformer](https://arxiv.org/abs/2012.09699)| Geometric learning | 2020 | A Generalization of Transformer Networks to Graphs |
| [GNN review](https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf)| Geometric learning | 2020 | Graph neural networks: A review of methods and applications |
| [GNN benchmark](https://arxiv.org/abs/2003.00982)| Geometric learning | 2020 | Benchmarking Graph Neural Networks |
| [Graphormer](https://arxiv.org/abs/2106.05234)| Geometric learning | 2021 | Graphormer: graph transformer |
| [Knowledge graph ](http://www.semantic-web-journal.net/system/files/swj2198.pdf)| Geometric learning | 2021 | On The Role of Knowledge Graphs in Explainable AI |
| [GNN explainaibility](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)| Geometric learning | 2017 | Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization |
| [GNN explainaibility](https://ieeexplore.ieee.org/document/8354201)| Geometric learning | 2018 | Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks |
| [GNN explainaibility](https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)| Geometric learning | 2019 | GNNExplainer: Generating Explanations for Graph Neural Networks |
| [GNN explainaibility](https://arxiv.org/abs/1909.10911)| Geometric learning | 2019 | Layerwise Relevance Visualization in Convolutional Text Graph Classifiers |
| [GNN explainaibility](https://arxiv.org/abs/1905.13686)| Geometric learning | 2019 | Explainability Techniques for Graph Convolutional Networks |
| [GNN explainaibility](https://arxiv.org/abs/2001.06216)| Geometric learning | 2020 | GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks |
| [GNN explainaibility](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8285761/)| Geometric learning | 2021 | Deep Graph Mapper: Seeing Graphs Through the Neural Lens |



&nbsp;

# Books

I am listing here free books or books that you can access for free from the provided links. I am listing books that are related to statistics, machine learning and artificial intelligence or connected in general to data science.

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [Goodfellow and Bengio](https://www.deeplearningbook.org/) | deep learning | 2016 | One of the most famous book about deep learning|
| [Bishop](https://readyforai.com/download/pattern-recognition-and-machine-learning-pdf/) | Machine learning | 2018 | Pattern Recognition and Machine Learning |
| [Bronstein, Bruna, Cohen and Veličković](https://arxiv.org/abs/2104.13478) | Geometric learning | 2021 | Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges |

&nbsp;

# Tools

A list of some useful tools that can be needed in different contexts. 

| Link | description | 
| --------- | ------ | 
|  [NN drawer](http://alexlenail.me/NN-SVG/LeNet.html) | Draw online your neural network architechture | 
|  [NN in Latex](http://alexlenail.me/NN-SVG/LeNet.html) | Latex code for drawing neural networks for reports and presentation | 
|  [Graphs drawer](https://graphonline.ru/en/) | tool to draw graph online | 
|  [Diagrams in Python](https://diagrams.mingrammer.com/) | library to draw diagrams in python | 

&nbsp;

# Dataset database

Here a list of many dataset database/repository for many different fields of artificial intelligence. For a ML/AI task you need two things: data and algorithm. This list is meant to help you witht he first part of the question: the data. I am providing here a list of links to databases where you can find many available to datasets (check the dataset section in the repository where I provide single datasets).

## Miscellaneous

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| Kaggle | [link](https://www.kaggle.com/) | A great source for datasets, with code and competitions. |
| UCI machine learning | [link](https://archive.ics.uci.edu/ml/index.php) | One of the oldest repository, user contributed. Not all the dataset are clean, but you can download without  |
| School System Finance | [link](https://archive.ics.uci.edu/ml/index.php) | One of the oldest repository, user contributed. Not all the dataset are clean, but you can download without  |
| EU Open Data Portal | [link](https://data.europa.eu/en) | More than a 1 million datasets released by the EU about health, finance, science etc... |
| Data. gov | [link](https://www.data.gov/) | Data from various US government departments, but it can tricky to exploit the data |
| Google Dataset Search | [link](https://datasetsearch.research.google.com/) | Free to search, but does include some fee-based search results.  is like Google’s standard search engine, but strictly for data |
| Earth Data | [link](https://earthdata.nasa.gov/) | this repository provides access to all of NASA’s satellite observation data for our little blue planet. |
| CERN Open Data Portal| [link](https://earthdata.nasa.gov/) | Physics data, over two petabytes of information, including datasets from the Large Hadron Collider particle accelerator |
| CERN Open Data Portal| [link](http://opendata.cern.ch/) | Physics data, over two petabytes of information, including datasets from the Large Hadron Collider particle accelerator |
| BFI film industry statistics | [link](https://www.bfi.org.uk/industry-data-insights) | Data on everything from UK box office figures, to audience demographics, home entertainment, movie production costs, and more |
| FBI Crime Data Explorer | [link](https://crime-data-explorer.fr.cloud.gov/pages/home) |  a broad collection of crime statistics from a variety of state organizations (universities and local law enforcement) and government (on a local, regional, and state-level). There are also guides about the data |
| Reddit discussion | [link](https://www.reddit.com/r/opendata/) | a discussion about open dataset on Reddit |
| FiveThirtyEight  | [link](https://data.fivethirtyeight.com/) |  best known for  data journalism,  the site also makes most of the data it uses in its reporting open to the public (mostly on politics, sports and culture)|
| Pew Internet | [link](https://www.pewresearch.org/internet/datasets/) |  data repository with a major focus on culture and media, social media, media consumption. |
| Data world | [link](https://www.pewresearch.org/internet/datasets/) |  one of the largest collection of open datasets, there are datasets in a large range of categories  |
| Papers with code| [link](https://paperswithcode.com/datasets) | around 5000 datasets about NLP, computer visions and so on|

&nbsp;

## Finance and economics

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| Quandl | [link](https://www.data.gov/) | Great source of economic and financial data |
| Word Bank | [link](https://data.worldbank.org/) | You can download data about population demographics, global economic and development indicators |
| IMF statistics | [link](https://www.imf.org/en/Data) | International Monetary Fund data about international finaces, debt rate, etc... |
| American Economic Association  | [link](https://data.worldbank.org/) | US macroeconomic data concerning employment, economic output, and other  variables |
| Google Trends| [link](https://trends.google.com/trends/?q=google&ctab=0&geo=all&date=all&sort=0) | Internet search activity details and  trends |
| Datahub.io| [link](https://datahub.io/collections) | Mostly free, no registration required. It covers a variety of topics from climate change to entertainment, it mainly focuses on areas like stock market data, property prices, inflation, and logistics. Data are updated frequently (monthly)|

&nbsp;

## Health and biology

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| Health data| [link](https://healthdata.gov/) | health data, including recent datasets for Covid-19, collected from the U.S. Department of Health and Human Services |
| Human Mortality| [link](https://www.mortality.org/) | information from 41 different countries, this dataset provides detailed mortality and population data |
| Big city| [link](https://www.mortality.org/) | Big Cities Health Coalition's upgraded data platform allows comparisons of key public health indicators across 28 large, urban cities|
| National Library of Medicine| [link](https://datadiscovery.nlm.nih.gov/) | BData Discovery is a platform providing access to datasets from selected NLM resources.  Users can explore, filter, visualize, and export data in a variety of formats, including Excel, JSON, XML, as well as access and build with these datasets via API.|
| National Cancer Institute SEER Data| [link](https://seer.cancer.gov/data-software/) | The Surveillance, Epidemiology, and End Results Program offers population data by age, sex, race, year of diagnosis, and geographic areas|
| ELVIRA Biomedical Data Set Repository| [link](https://leo.ugr.es/elvira/DBCRepository/) | This is an online repository of high-dimentional biomedical data sets taking from the Kent Ridge Biomedical Data Set Repository,  including gene expression data, protein profiling data and genomic sequence data that are related to classification and that are published recently in Science, Nature and so on prestigious journals|
| MedPix| [link](https://medpix.nlm.nih.gov/home) | MedPix is a free open-access online database of medical images, teaching cases, and clinical topics, integrating images and textual metadata including over 12,000 patient case scenarios, 9,000 topics, and nearly 59,000 images. |
| WHO| [link](https://www.who.int/data/collections) | The World Health Organization manages and maintains a wide range of data collections related to global health and well-being as mandated by its Member States. |
| CDC| [link](https://www.cdc.gov/datastatistics/index.html) | the CDC offer a large list of datasetsabout health diseases. |
| Data Carpentry for Biologists | [link](https://datacarpentry.org/semester-biology/materials/datasets/) | A list of datasets for biology that you can download from the website (in different format csv, excel, zip) |
| Cancer Image Archive | [link](https://www.cancerimagingarchive.net/) | A large archive of medical images of cencer that are accessible for public download |
| Image Data Resource | [link](http://idr.openmicroscopy.org/about/) | The Image Data Resource (IDR) is a public repository of reference image datasets from published scientific studies. IDR enables access, search and analysis of these highly annotated datasets. |
| SICAS Medical Image | [link](https://www.smir.ch/) | The SICAS Medical Image Repository host datasets of medical images that you can browse (or where you can store your dataset) |

&nbsp;

## Relational and network analysis

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| LINQS| [link](https://linqs.soe.ucsc.edu/data) | different relational dataset for graph analysis |
| PyTorch Geometric| [link](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) | many different relational dataset for graph analysis |
| networkrepository | [link](https://networkrepository.com/index.php) | interactive data and network data repository with real-time visual analytics, comprehensive collection of network graph data in 30+ domains (network science, bioinformatics, machine learning, data mining, physics, and social science) |
| Network data | [link](http://www-personal.umich.edu/~mejn/netdata/) | a list of network data that can be downloaded |
| Group Lens| [link](https://grouplens.org/datasets/movielens/) | Different rating data sets from the MovieLens web site collected over various periods of time and with different size |

&nbsp;

## Computer vision

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| CV Datasets on the web| [link](http://www.cvpapers.com/datasets.html) | a large list of dataset for computer visions |
| Yet Another Computer Vision Index To Datasets (YACVID)| [link](http://yacvid.hayko.at/) | a large list of dataset for computer visions, but not all the links work well|
| TU Darmstadt dataset| [link](https://www.visinf.tu-darmstadt.de/vi_research/datasets/index.en.jsp) | a list of dataset collected by TU Darmstadt  university |
| CVonline| [link](https://homepages.inf.ed.ac.uk/rbf/CVonline/CVentry.htm) | Another list of dataset for computer vision |
| COVE| [link](https://cove.thecvf.com/) | COVE is an online repository for computer vision datasets sponsored by the Computer Vision Foundation. It is intended to aid the computer vision research community and serve as a centralized reference for all datasets in the field |
| Roboflow| [link](https://public.roboflow.com/) | Roboflow hosts free public computer vision datasets in many popular formats (including CreateML JSON, COCO JSON, Pascal VOC XML, YOLO v3, and Tensorflow TFRecords). |


&nbsp;

## Natural Language processing

| Repository | Link |  Description |
| ------- | ----------- | ------ |
| The Big Bad NLP Database| [link](https://index.quantumstat.com/) | a database for NLP containing more than 300 dataset. Most of them are in english, only few in other languages. |
| Curated NLP Database | [link](https://metatext.io/datasets) | List of 1000+ Natural Language Processing Datasets. Covering tasks from classification to question answering, languages from English, Portuguese to Arabic |
| nlp-datasets | [link](https://metatext.io/datasets) | Github repository,  list of free/public domain datasets with text data for use in Natural Language Processing (NLP).|
| NLTK Corpora| [link](http://www.nltk.org/nltk_data/) | NLTK has built-in support for dozens of corpora and trained models|
| Hugging Face datasets| [link](https://huggingface.co/datasets) | nearly 3000 datasets focused mainly on NLP |
| airXiv | [link](https://arxiv.org/help/bulk_data_s3) |  arXiv bulk data available for download (270 GB of text), updated almost each month |


&nbsp;

# Free Courses

A list of free available courses, the list is not exhaustive. I am not listing Coursera or other MOOC courses (which you can find on the platforms).

| Name | Link |  Description |
| ------- | ----------- | ------ |
| OpenIntro Statistics | [link](https://www.openintro.org/book/os/) | a source for statistics |
| UVA deep learning | [link](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) | Deep learning with PyTorch |

&nbsp;

# Miscellaneous

Other interesting resources

| Name | Link |  Description |
| ------- | ----------- | ------ |
|  The Super Duper NLP Repo | [link](https://notebooks.quantumstat.com/) | a collection of Colab notebooks covering a wide array of NLP task implementations available to launch in Google Colab with a single click. |


# Contributing

You can use the issue section or you can write to me. If there are interesting resources, feel free to suggest I will add as soon as I can.

# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
