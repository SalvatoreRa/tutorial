# Additional Resources
## Resources on the artfificial intelligence

![Resources](https://images.unsplash.com/photo-1548048026-5a1a941d93d3?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80)

Photo by [Daniel](https://unsplash.com/@setbydaniel) on [Unsplash](https://unsplash.com/)

&nbsp;

In this section, I will suggest and add many resources on artificial intelligence that can be useful. The list is not exhaustive and I will expand with time, if you want to suggest other resources, you are welcome.

I am listing seminal articles (the list is clearly not exhaustive), free available books and useful tools that I found.

&nbsp;

# **Table of Contents**

* Scientific articles
* Books
* Tools
* Repositories
* Free courses


&nbsp;

# Scientific articles

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
| [Word2vec](https://arxiv.org/pdf/1301.3781v3.pdf)| NLP | 2013 | Word2vec, word embedding |
| [Layer normalization](https://arxiv.org/abs/1607.06450)| NLP | 2013 | layer normalization |
| [RNN ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2015 |Visualizing and Understanding Recurrent Networks |
| [LSTM ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2015 |Review about LSTM |
| [GRU ](https://arxiv.org/pdf/1506.02078v1.pdf)| NLP | 2014 |Gated recurrent unit |
| [Transformers ](https://arxiv.org/abs/1706.03762)| NLP | 2016 |First paper proposing transformers |
| [BERT ](https://arxiv.org/abs/1810.04805)| NLP | 2018 | First paper proposing BERT |
| [DistilBERT ](https://arxiv.org/abs/1910.01108)| NLP | 2019 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter |
| [Transformers ](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)| NLP | 2018 |GPT-1 paper |
| [Transformers ](https://openai.com/blog/better-language-models/)| NLP | 2018 |GPT-2 paper |
| [Transformers ](https://arxiv.org/abs/2005.14165)| NLP | 2020 |GPT-3 paper |
| [Attention reivew ](https://arxiv.org/abs/1904.02874)| NLP | 2019 | An Attentive Survey of Attention Models |
| [GAN ](https://arxiv.org/pdf/1406.2661v1.pdf)| Generative Learning | 2014 |Generative Adversarial Nets |
| [GAN ](https://arxiv.org/abs/1511.06434)| Generative Learning | 2015 |original article presenting DC-GAN |
| [ADAM](https://jmlr.org/papers/v15/srivastava14a.html)| General interest | 2014 |first paper presenting ADAM |
| [Group normalization](https://arxiv.org/pdf/1803.08494.pdf)| General interest | 2018 | discussing group normalization |
| [Dropout](https://arxiv.org/abs/1412.6980)| General interest | 2014 |first paper on dropout |
| [System design](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)| General interest | 2015 | Hidden Technical Debt in Machine Learning Systems |
| [Tabular data](https://arxiv.org/abs/2106.03253)| Tabular data | 2021 |Tabular Data: Deep Learning is Not All You Need (showing that deep learning with tabular data is still not the first choice) |
| [Tabular data](https://arxiv.org/pdf/2106.01342.pdf)| Tabular data | 2021 |Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training |
| [Tabular data](https://arxiv.org/pdf/2110.01889.pdf)| Tabular data | 2021 | Method's review on Deep Neural Networks and Tabular Data |
| [Tabular data](https://arxiv.org/pdf/2106.11189.pdf)| Tabular data | 2021 | how regularization impact Neural Network for tabular data|
| [RL](https://arxiv.org/abs/1312.5602)| Reinforcement Learning | 2013 | Playing Atari with Deep Reinforcement Learning |
| [GNN](https://arxiv.org/abs/1611.08097)| Geometric learning | 2016 | Geometric deep learning: going beyond Euclidean data |
| [GNN](https://arxiv.org/abs/1609.02907)| Geometric learning | 2016 | Semi-Supervised Classification with Graph Convolutional Networks |
| [Knowledge graph review](https://iopscience.iop.org/article/10.1088/1742-6596/1487/1/012016)| Geometric learning | 2020 | A Survey on Application of Knowledge Graph |
| [Graph transformer](https://arxiv.org/abs/2012.09699)| Geometric learning | 2020 | A Generalization of Transformer Networks to Graphs |
| [GNN review](https://arxiv.org/ftp/arxiv/papers/1812/1812.08434.pdf)| Geometric learning | 2020 | Graph neural networks: A review of methods and applications |
| [GNN benchmark](https://arxiv.org/abs/2003.00982)| Geometric learning | 2020 | Benchmarking Graph Neural Networks |
| [Graphormer](https://arxiv.org/abs/2106.05234)| Geometric learning | 2021 | Graphormer: graph transformer |
| [Knowledge graph ](http://www.semantic-web-journal.net/system/files/swj2198.pdf)| Geometric learning | 2021 | On The Role of Knowledge Graphs in Explainable AI |

&nbsp;

# Books

| Link | Topic | Year | Description |
| --------- | ------ | ------ |------ |
| [Goodfellow and Bengio](https://www.deeplearningbook.org/) | deep learning | 2016 | One of the most famous book about deep learning|
| [Bishop](https://readyforai.com/download/pattern-recognition-and-machine-learning-pdf/) | Machine learning | 2018 | Pattern Recognition and Machine Learning |
| [Bronstein, Bruna, Cohen and Veličković](https://arxiv.org/abs/2104.13478) | Geometric learning | 2021 | Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges |

&nbsp;

# Tools

| Link | description | 
| --------- | ------ | 
|  [NN drawer](http://alexlenail.me/NN-SVG/LeNet.html) | Draw online your neural network architechture | 
|  [NN in Latex](http://alexlenail.me/NN-SVG/LeNet.html) | Latex code for drawing neural networks for reports and presentation | 
|  [Graphs drawer](https://graphonline.ru/en/) | tool to draw graph online | 
|  [Diagrams in Python](https://diagrams.mingrammer.com/) | library to draw diagrams in python | 

&nbsp;

# Repositories

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
| Health data| [link](https://healthdata.gov/) | health data, including recent datasets for Covid-19, collected from the U.S. Department of Health and Human Services |
| Human Mortality| [link](https://www.mortality.org/) | information from 41 different countries, this dataset provides detailed mortality and population data |
| Big city| [link](https://www.mortality.org/) | Big Cities Health Coalition's upgraded data platform allows comparisons of key public health indicators across 28 large, urban cities|
| National Library of Medicine| [link](https://datadiscovery.nlm.nih.gov/) | BData Discovery is a platform providing access to datasets from selected NLM resources.  Users can explore, filter, visualize, and export data in a variety of formats, including Excel, JSON, XML, as well as access and build with these datasets via API.|
| National Cancer Institute SEER Data| [link](https://seer.cancer.gov/data-software/) | The Surveillance, Epidemiology, and End Results Program offers population data by age, sex, race, year of diagnosis, and geographic areas|

&nbsp;

# Free Courses

| Name | Link |  Description |
| ------- | ----------- | ------ |
| OpenIntro Statistics | [link](https://www.openintro.org/book/os/) | a source for statistics |
| UVA deep learning | [link](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) | Deep learning with PyTorch |


# Contributing

You can use the issue section or you can write to me. If there are interesting resources, feel free to suggest I will add as soon as I can.

# License

This project is licensed under the **MIT License** 

# Bugs/Issues

Comment or open an issue on Github
