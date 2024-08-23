# Frequently Asked Questions (FAQs) on machine learning and artificial intelligence

![artificial intelligence](https://github.com/SalvatoreRa/tutorial/blob/main/images/nn_brain.jpeg?raw=true)

Photo by [Alina Grubnyak](https://unsplash.com/@alinnnaaaa) on [Unsplash](https://unsplash.com/)

&nbsp;

This is a collection of FAQs and their answers about ML, AI, neural networks, and LLM. Many of these questions have been asked to me by students or other ML practitioners, I decided to collect and discuss them here.

The list is on construction and I am expanding it.

# Index
* [FAQ on machine learning](#FAQ-on-machine-learning)
* [FAQ on artificial intelligence](#FAQ-on-artificial-intelligence)

&nbsp;

# FAQ on machine learning

## Statistics and data science

<details>
  <summary><b>What is better Pandas or Polars? </b></summary>

Python is a great ecosystem for data science, on the other hand, some basic analysis becomes complex as the datasets grow (parallel processing, query optimization, and lazy evaluation) especially if you use pandas. **[Pandas](https://pandas.pydata.org/)** in fact it is single-threaded (so it means it runs on only one CPU), requires the entire dataset to be in memory, also it does not allow the order of operation optimization (it means the operations are sequential in the order of declaration which is not always the best choice). 

**[Polars](https://pola.rs/)** on the other hand offers: 
* **Parallel computing**, so use all available cores.
* **Improve storage** since it leverages Apache Arrow.
* **File scanning** which allows you to work with very large files without necessarily keeping the whole file in memory

**Polars** also is similar to Pandas in syntax so switching between libraries is fairly easy. Since there is a larger ecosystem compatible with Pandas, it is recommended to use it for small/medium datasets and use Polars for very large datasets
  
</details>

<details>
  <summary><b>Which correlation should use? </b></summary>

  There are different types of correlation, The most famous of which is the **Pearson correlation**. The correlation coefficient represents the linear relationship between two variables. Pearson correlation has this formula:

$$r_{XY} = \frac{\sum (X_i - \overline{X})(Y_i - \overline{Y})}{\sqrt{\sum (X_i - \overline{X})^2 \sum (Y_i - \overline{Y})^2}}$$


Where X and Y are the two variables, and $\overline{X}$ and $\overline{Y}$ represent the means. 

**Spearman correlation** is another popular alternative:

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

where $d_i$ represents the distance represents the difference between the ranks of corresponding values $X_i$ and $Y_i$.


The main differences between the two correlations soo: 
* Pearson measures linear relationships, Spearman correlation between variables that have a monotonic relationship. Pearson assumes that variables are normally distributed.
* As seen Pearson is based on covariance and the other is based on ranked data. However, both have ranges between -1 and 1.
* Pearson is more sensitive to outlier data. Person is more recommended for interval and ratio data, while Spearman is for ordinal and non-normally distributed data. 


As seen Pearson is recommended for linear relationships, while Spearman is recommended for monotonic associations. There is also **Kendall correlation**, but basically it is very similar to Spearman for assumptions. Linear relations are a special case of monotonic functions. A monotonic relation is where there is no change in direction or always increasing or always decreasing (not necessarily linearly)

![correlation relationship](https://github.com/SalvatoreRa/tutorial/blob/main/images/correlation_relation.webp?raw=true)
*from [here](https://www.quora.com/How-do-I-know-if-the-scatter-plot-displays-a-linear-a-monotonic-or-a-non-monotonic-relationship)*

This means, however, that there are cases where there is an association between two variables (neither linear nor monotonic) that none of these three types of correlation can detect

In a 2020 study, they propose a new relationship, which measures how much Y is a function of X (rather than whether there is a monotonic or linear relationship between the two). This new correlation is also based on ranking é has two possible formulas, one based on ties between the two variables and whether there are no ties (or probable no ties) between the variables. The first:

![correlation relationship](https://github.com/SalvatoreRa/tutorial/blob/main/images/correlation_no_ties.webp?raw=true)
*from [here](https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2004922)*

and if there are ties:

![correlation relationship](https://github.com/SalvatoreRa/tutorial/blob/main/images/correlation_ties.webp?raw=true)
*from [here](https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2004922)*

As can be seen, the new correlation method is not affected by the direction of the relationship (the range is between 0 and 1, with one being the maximum of the relationship). Where Pearson concludes that there is no relationship (for example, in the parabolic or sinusoidal case) this new method succeeds instead in showing a relationship.

![correlation relationship](https://github.com/SalvatoreRa/tutorial/blob/main/images/A%20New%20Coefficient%20of%20Correlation.png?raw=true)
*Values of ξn(X,Y)for various kinds of scatterplots, withn=100. Noise increases from left to right. from [here](https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2004922)*



If you want to try in Python the code:

```Python
from numpy import array, random, arange

def xicor(X, Y, ties=True):
    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
        return 1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l)))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return 1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1)

```

Others have noticed that also this correlation is not exempt from issues, for that reasons they suggest **mutual information-based coefficient R**:
* **non-linearity.** The xicorr does not capture all the types of non-linearities (like donuts).
* **symmetry.** The correlation should be symmetric (ρ(x,y)=ρ(y,x)), this is true for Pearson and R but not xicorr
* **Consistency.** In all the cases xicorr is consistent.
* **Scalability.** R is more scalable with the increase of data points
* **Precision.** R is more precise (precision is defined as stdev(A)/mean(A), meaning the variance should be small)

In any case, you can test these different correlations. While they are not present in a standard package in Pythons, I have collected them in a Python script you can easily import ([check here](https://github.com/SalvatoreRa/tutorial/tree/main/machine%20learning/utility))

Suggested lecture:
* [Myths About Linear and Monotonic Associations: Pearson’s r, Spearman’s ρ, and Kendall’s τ](https://www.tandfonline.com/doi/full/10.1080/00031305.2021.2004922)
* [A New Coefficient of Correlation](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1758115)

</details>


## Machine learning in general

<details>
  <summary><b>What is machine learning? </b></summary>

**Machine Learning** is the field of study that deals with learning or predicting something from data. In the traditional paradigm, you had to write hard-coded rules. A machine learning algorithm should infer these rules on its own. 

*"Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions" - from [Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)*

As an example, we want to create a model to classify sentimental analysis of movie reviews. A traditional approach would be to create rules (something like if/then, for example, if "amazing" is present as a word the review is positive). A machine learning approach instead takes a dataset with labeled elements (a set of reviews that have already been annotated to be positive or negative) and derives the rules on its own.

In some cases, these rules can be displayed. In this case, the model has learned boundaries to separate the various classes in the iris dataset:

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/sphx_glr_plot_voting_decision_regions_001.png?raw=true)
*from [here](https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html)*
  
</details>

<details>
  <summary><b>What is the central limit theorem? </b></summary>

*"In probability theory, the central limit theorem (CLT) states that, under appropriate conditions, the distribution of a normalized version of the sample mean converges to a standard normal distribution. This holds even if the original variables themselves are not normally distributed. " - [source](https://en.wikipedia.org/wiki/Central_limit_theorem)* 

**central limit theorem (CLT)** in a nutshell says that the distribution of sample means approximates a normal distribution as you increase the sample size. It is one of the most fundamental statistical theorems and is an important assumption for many algorithms. In other words. A key aspect is that the average of the sample means will be equal to the true population mean and standard deviation. In other words, with a sufficiently large sample size, we can predict the characteristics of a population
  
</details>

<details>
  <summary><b>What is overfitting? How to prevent it? </b></summary>
  
  **overfitting** is one of the most important concepts in machine learning, usually occurring when the model is too complex for a dataset. The model then tends to learn patterns that are only present in the training set and thus not be able to generalize effectively. So it will not have adequate performance for unseen data.

**Underfitting** is the opposite concept, where the model is too simple and fails to generalize because it has not identified the right patterns. With both overfitting and underfitting, the model is underperforming

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/under_overfitting.svg?raw=true)
*from Wikipedia*

Solutions are usually: 
* to collect more data when possible.
* Eliminate features that are not needed.
* Start with a simple model such as logistic regression as a baseline and compare it with progressively more complex models.
* Add regularization techniques. Use ensembles. 

</details>

<details>
  <summary><b>When to use K-fold cross-validation or group K-fold? </b></summary>

**K-fold cross-validation** is one of the most widely used evaluation methods for a machine learning model. It is usually used to understand how a model behaves when there is unseen data. K-fold cross-validation is simple, we have a dataset X and a target variable y. The dataset is divided into K folds (then a subset of X and y) and for each interaction we train the model on k-1 fold and calculate the error on the remaining fold. If we have 100 examples and k =5, it means that at each iteration we select 20 random examples, train the model on the other 80 examples, and calculate the performance on the 20 examples.
  
  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/1024px-K-fold_cross_validation_EN.svg.png?raw=true)
*from Wikipedia*

The main problem with k-fold cross-validation is that we assume that all the different folds have the same distribution. This is not true in a number of cases where the dataset is stratified by an additional temporal, group, or spatial dimension. This causes a so-called information leak and is easily understood when we look at data that are temporally stratified. If we use random shuffling, the model will see into the future and we have what can be called data leakage

cross-validation leads to predictions that are overly optimistic (overly confident), favors models that are prone to overfitting. So for real-world cases, we need an alternative that avoids leakage between folds. This can be achieved with **group folds**:

*"GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing and training sets. For example if the data is obtained from different subjects with several samples per-subject and if the model is flexible enough to learn from highly person specific features it could fail to generalize to new subjects. GroupKFold makes it possible to detect this kind of overfitting situations." -[source](https://scikit-learn.org/stable/modules/cross_validation.html)*

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/sphx_glr_plot_cv_indices_007.png?raw=true)
*from scikit-learn*

</details>

<details>
  <summary><b>Should I use Class imbalance corrections?</b></summary>
  
  In general, there are plenty of methods for correcting class imbalance data, though it is not always a good idea to do so. **Imbalance data** is when in a classification dataset there is an overabundance of one of the class labels. In this case, the most abundant class is called majority class and the other minority class (this is in the case of binary classification, but class imbalance can also occur in the case of multiclass classification).

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/imbalance_data.png?raw=true)
*from [here](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)*

Generally, the most commonly used strategies are: 
* **Downsample the majority class.** in this case the goal is to reduce the number of examples in the majority class to have a balanced dataset (but we risk losing information).
* **Upsampling of the minority class.** conversely we increase the number of examples in the minority class by exploiting machine learning or artificial intelligence approaches (this can introduce bias though)

For example, when we are interested in a well-calibrated model, oversampling may do more harm than not. A calibrated model is when we can interpret he output of such a model in terms of a probability. We might actually think that all models are calibrated, but in general models are overconfident (for example in the case of binary classification, an overconfident model predicts values close to 0 and 1 in many cases where they should not do). 

*"Model calibration captures the accuracy of risk estimates, relating to the agreement between the estimated (predicted) and observed number of events. In clinical applications where a patient’s predicted risk is the entity used to inform clinical decisions, it is essential to assess model calibration. If a model is poorly calibrated, it may produce risk estimates that do not approximate a patient’s true risk well [3]. A poorly calibrated model may produce predicted risks that consistently over- or under-estimate true risk or that are too extreme (too close to 0 or 1) or too modest (too close to event prevalence)"* -- from [here](https://arxiv.org/pdf/2404.19494)

A simpler example, if we have a model that predicts the probability of a fire in a building, a model calibrated when it gives a probability of 0.8 or 0.2, means that in the first case a fire is four times more likely. In the case of an uncalibrated model these probabilities do not have the same meaning.

*"Overall, as imbalance between the classes was magnified, model calibration deteriorated for all prediction models. All imbalance corrections affected model calibration in a very similar fashion. Correcting for imbalance using pre-processing methods (RUS, ROS, SMOTE, SENN) and/or by using an imbalance correcting algorithm (RB, EE) resulted in prediction models which consistently over-estimated risk. On average, no model trained with imbalance corrected data outperformed the control models in which no imbalance correction was made, with respect to model calibration."* -- from [here](https://arxiv.org/pdf/2404.19494)

In other words, when we are interested in a calibrated model, dataset-balancing techniques do more harm than good.

This is then in line with early reports that showed that SMOTE works only with weak learners and is destructive of model calibration.

In fact, you rarely see it used on Kaggle (or at least it does not seem to be a winning strategy). According to this stems from the fact that models like SMOOTH implicitly assume that the class distribution is sufficiently homogenous around the minority class instances. Which is then not necessarily the case (especially since not all variables are so homogenous).

**So we should never use it?**

According to this [article](https://arxiv.org/pdf/2201.08528) as a general rule: 
* Balancing could improve prediction performance for weak classifiers but not for the SOTA classifiers. The strong classifiers (without balancing) yield better prediction quality than the weak classifiers with balancing.
* For label metric (example F1) optimizing the decision threshold is recommended do to simplicity and lower compute cost (nowhere is it written that a threshold of 0.5 must be used necessarily).

Suggested lecture:
* [The harms of class imbalance corrections for machine learning based prediction models: a simulation study](https://arxiv.org/abs/2404.19494)
* [To SMOTE, or not to SMOTE?](https://arxiv.org/abs/2201.08528)
* [Are unbalanced datasets problematic, and (how) does oversampling (purport to) help?](https://stats.stackexchange.com/questions/357466/are-unbalanced-datasets-problematic-and-how-does-oversampling-purport-to-he)
* [Why SMOTE is not used in prize-winning Kaggle solutions?](https://datascience.stackexchange.com/questions/106461/why-smote-is-not-used-in-prize-winning-kaggle-solutions)
  
</details>

<details>
  <summary><b>What is gradient descent? What are the alternatives?</b></summary>
  !
</details>

## Clustering

<details>
  <summary><b>Should I use the elbow method?</b></summary>

  The **elbow method** has long been the most widely used method to evaluate the clustering number for **k-means**.  

Starting with a dataset, we create a loop in which we test an increasing number of clusters. The idea is simple, at each iteration of the k-means we calculate the inertia (sum of squared distances between each point and the center of the cluster it is assigned). the inertia goes down with the number of clusters because the clusters get smaller and smaller, and thus the points get closer and closer to the center of the cluster. Plotting the inertia on the y-axis and the number of clusters, there is a sweet spot that represents the point of maximum curvature (beyond this point increasing the number of clusters is no longer convenient).

There are many other methods for being able to analyze the number of clusters for an algorithm.
  
</details>


## Tree-based models

<details>
  <summary><b>What is bagging or boosting?</b></summary>

Bagging and boosting are two different ensemble techniques that are used to improve the performance of an ensemble of decision trees by reducing system error. The basic idea is that each individual decision tree is trained with a different dataset. The main difference is that bagging trains the different models on different subsets of data while boosting conducts the training sequentially, focusing on the error committed by the previous model. 

An ensemble is a machine learning technique in which we combine different models to improve performance. By combining different weak learners we get a model that has better performance. The disadvantage is that these models if not properly regularized can go into overfitting.

More in detail, **Bagging** combines multiple models that are trained on different datasets with the aim of reducing the variance of the system (by averaging the error of the different models that make up the ensemble). Given a dataset, different datasets are created for each ensemble decision tree (this is usually conducted by bootstrapping). For predictions, each model produces a prediction and the majority prediction is usually chosen. An example of this approach is Random Forest.

We then initially randomly conduct bootstrap sampling of the initial dataset (sampling with replacement) and train a single model on this subset. For each weak learner, this process is repeated. For classification, we combine predictions with majority voting (while for regression we average the predictions).

  ![bagging](https://github.com/SalvatoreRa/tutorial/blob/main/images/bagging.png?raw=true)
*from [here](https://arxiv.org/abs/2104.02395)*

In boosting, each model depends on the models that have been previously trained. This allows for a system that is better adapted to the dataset. Sampling of the data is conducted and then a weak learner (a tree) is trained on that data. Initially, for each sample in the dataset, we have the same weight. The error for each sample is then calculated, the greater the error the greater the weight that will be assigned to that sample. The data are passed to the next model. Each model also has an associated weight based on the goodness of its predictions. The model weight is used to conduct a weighted average over the final predictions. 

Boosting attempts to sequentially reduce the error of the models in the ensemble by trying to correct misclassifications of the previous model (thus reducing both bias and variance of the system). Examples of boosting algorithms are: AdaBoost, XGBoost, Gradient Boosting Mechanism. 

  ![boosting](https://github.com/SalvatoreRa/tutorial/blob/main/images/boosting.png?raw=true)
*from [here](https://arxiv.org/abs/2104.02395)*

Suggested lecture:
  * [Evolutionary bagging for ensemble learning](https://arxiv.org/pdf/2208.02400)
  * [Ensemble deep learning: A review](https://arxiv.org/abs/2104.02395)
  
  
</details>

<details>
  <summary><b>Why there are different impurity metrics?</b></summary>
  !
</details>

<details>
  <summary><b>Should I use XGBoost? Or Catboost, LightGBM, random forest?</b></summary>
  !
</details>

<details>
  <summary><b>Are tree-based models better than neural networks for tabular data?</b></summary>
  !
</details>

&nbsp;

# FAQ on artificial intelligence

<details>
  <summary><b>What are the differences between machine learning and artificial intelligence? </b></summary>

In essence, **artificial intelligence** is a set of algorithms and techniques that exploits neural networks to solve various tasks. These are neural networks composed of several layers that can extract complex features from a dataset. This makes it possible to avoid feature engineering and at the same time learn a complex representation of the data. Neural networks also can learn relationships that are nonlinear and thus complex patterns that other machine learning algorithms can hardly learn. This clearly means having patterns with many more parameters and thus the need for appropriate learning algorithms (such as backpropagation).
 
</details>

<details>
  <summary><b>What is supervised learning? self-supervised learning?  </b></summary>
  
  
**Supervised learning** uses labeled training data and **self-supervised learning** does not. In other words, when we have labeled we can use supervised learning, only sometimes we don't have it and for that, we need other algorithms.

In supervised learning we usually have a dataset that has labels (for example, pictures of dogs and cats), we divide our dataset into training and testing and train the model to solve the task. Because we have the labels we can check the model's responses and its performance. In this case, the model is learning a function that binds input and output data and tries to find the relationships between the various features of the dataset and the target variable. Supervised learning is used for classification, regression, sentiment analysis, spam detection, and so on.

In _unsupervised learning (or self-supervised)_, on the other hand, the purpose of the model is to learn the structure of the data without specific guidance. For example, if we want to divide our consumers into clusters, the model has to find patterns underlying the data without knowing what the actual labels are (we don't have them after all). These patterns are used for anomaly detection, big data visualization, customer segmentation and so on.

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/Supervised-and-unsupervised-machine-learning-a-Schematic-representation-of-an.png?raw=true)
*from [here](https://www.researchgate.net/figure/Supervised-and-unsupervised-machine-learning-a-Schematic-representation-of-an_fig3_351953193)*

**Semi-supervised learning** is an intermediate case, where we have few labels and a large dataset. The goal is to use the few labeled examples to label a larger amount of unlabeled data

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/semisup.webp?raw=true)
*from [here](https://medium.com/@gayatri_sharma/a-gentle-introduction-to-semi-supervised-learning-7afa5539beea)*

**self-supervised learning** is now stricly related to **transfer learning** (check below). In fact, many models are trained unsupervised on a huge amount of data. For example, Transformers are trained on a lot of textual data using a huge amount of data during the pretraining phase. During this phase, language modeling or another pretext task is used to make the model learn a knowledge of the language. Labeling this data would be too expensive, so the purpose of the model is to learn the structure of the language during pretraining. Only at a later stage is the model adapted for a specific task. So in this case our purpose is to take advantage of the amount of data and a task that allows us to train the model, without having to annotate or specify the task. 

</details>

<details>
  <summary><b>What is transfer learning? </b></summary>

**transfer learning** is a process in which we exploit a model's abilities for a different task than what was originally trained.

*"Transfer learning and domain adaptation refer to the situation where what has been learned in one setting … is exploited to improve generalization in another setting" -[source](https://www.deeplearningbook.org/)* 

*"Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned." -[source](https://dl.acm.org/doi/10.5555/1803899)* 


Transfer learning requires the model to learn features that are general. So they are usually models that are trained on a huge amount of data and can learn very different patterns from each other. 

For example, a large convolutional network such as ResNet is trained on a large number of images such as Imagenet, then the model is retrained to classify images of dogs or cats. In this case, the model head that is specific to the original task (imagenet) is removed and replaced with a final layer for the specific task.

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/Transfer_learning.svg.png?raw=true)
*from [here](https://en.wikipedia.org/wiki/Transfer_learning)*

Another widely used case is the transformer. The transformer is a very large model that can learn a large number of patterns. In this case, the pattern is not trained for a specific task but in self-supervised learning. This first phase is called the pre-training phase. During this initial phase, the model learns language features by predicting the next word in a word sequence (or as a masked language model, where some words are masked and the model has to predict them). Once this is done, the model can be repurposed for other tasks such as sentiment analysis (a classification task).

This approach can be used with so many types of data, for example for images, you can mask part of the images and the model has to predict what is in the masked patch.
  
</details>

<details>
  <summary><b>What is knowledge distillation? </b></summary>
  LLMs are getting bigger and bigger, reaching even more than 100B parameters. These models excel in different tasks and achieve state-of-the-art in all benchmarks. But do we need an LLM for every task? 

Sometimes we need a model that is capable of accomplishing a task with as much accuracy as possible but is also computationally efficient (e.g., a classification task that must run on a device). For this a smaller model might be fine, the important thing is that it is capable of doing the task as well as possible. The idea behind **Knowledge Distillation** is that we can distill a complex model into a smaller, more efficient model that needs limited resources. The idea behind it is that the larger model is a "teacher" and the smaller model is a "student." In practice we use the soft probabilities (or logits) of the teacher network to supervise the student network along with the class labels, this is because these probabilities provide more information than a label alone and allow the model to learn better

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/knowledge_distillation.png?raw=true)
*from [here](https://arxiv.org/pdf/2006.05525.pdf)*

To give a more concrete example, we have a model like ResNet-50 that is trained on millions of images of a thousand different classes, but we need a model that can recognize dogs and cats (two classes) and has few layers. The idea is to create a model (student) that is able to mimic the generalization ability of the more complex model (i.e. ResNet) and we use the probabilities generated by the teacher network to train it. The advantage is that we generally need much less data than training the student model from scratch and without a teacher

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/knowledge_distillation2.png?raw=true)
*from [here](https://arxiv.org/pdf/2006.05525.pdf)*

So we have a teacher model (ResNet in our example) that generates probabilities for each class. After that we take the student model and train it for the same data, again obtaining a probability distribution, exploiting a distillation loss we try to make these probability distributions similar to those of the teacher. In addition, we have a cross-entropy loss in which we use the actual labels of the data. The student model is trained by exploiting these two losses so it learns from both the teacher model and the real labels.

Suggested lecture:
  * [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
  

</details>



## Neural networks

<details>
  <summary><b>What is an artificial neuron?</b></summary>

  A **neural network** is a collection of artificial neurons. The **artificial neuron** is clearly inspired by its human counterpart. The first part of the biological neuron receives information from other neurons. If this signal is relevant gets excited and the potential (electrical) rises, if it exceeds a certain threshold the neuron activates and passes the signal to other neurons. This transfer is passed through the axon and its terminals that are connected to other neurons.

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/neuron.png?raw=true)
*from Wikipedia*

  This is the corresponding equation for the artificial neuron:
  
$$\[y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)\]$$

As we can see, this equation mimics the behavior of the human neuron. X inputs are the signals coming from other neurons, the neuron weighs their importance by multiplying them with a set of weights. Once this information is weighed, this sum is called the transfer function. If the information is relevant it must pass a threshold, in this case given by the activation function. If it is above the threshold, the neuron is activated and passes the information (in the biological one this is called firing). This becomes the input for the next neuron.

</details>

<details>
  <summary><b>What are neural networks?</b></summary>
  
  In a nutshell, **Neural networks**  are a series of layers composed of different artificial neurons. We have an input layer, several hidden layers, and an output layer. The first layer takes inputs and is therefore specific to inputs. The hidden layers learn a representation of the data, while the last layer is specific to the task (classification, regression, auto-encoder, and so on).

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/ann.webp?raw=true)
*from [here](https://www.cloudflare.com/learning/ai/what-is-neural-network/)*

We generally distinguish: 
* **shallow neural networks**, where you have one or few hidden layers
* **Deep neural networks**, where you have many hidden layers (more about below)

Neural networks despite having many parameters have the advantage of extracting sophisticated and complicated representations from data. They also learn high-level features from the data that can then be reused for other tasks. An additional advantage is that neural networks generally do not require complex pre-processing like traditional machine learning algorithms. Neural networks were invented with the purpose, that models would learn features on their own even when the data is noisy (a kind of automatic "feature engineering")

</details>

<details>
  <summary><b>What is a Convolutional Neural Network (CNN)?</b></summary>

**Convolutional neural networks** are a subset of neural networks that specialize in image analysis. These neural networks are inspired by the human cortex, especially the visual cortex. The two main concepts are the creation of a hierarchical representation (increasingly complex features from the top to the bottom of the network) and the fact that successive layers have an increasingly larger receptive field size (layers further forward in the network see a larger part of the image).

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/cnn1.png?raw=true)

This causes convolutional neural networks to create an increasingly complex representation, where layers further down the network recognize simpler features (edges, textures) and layers further up the network recognize more complex features (objects or faces)

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/cnn2.png?raw=true)

How does it actually work?

Pixels that are close together represent the same object and pattern, so they should be processed together by the neural network. These neural networks consist of three main layers: a convolutional layer to extract features. Pooling layer, to reduce the spatial dimension of the representation. Fully connected layer, usually the last layers to map the representation between input and output (for example, if we want to classify various objects).

A convolutional layer basically accomplishes the dot product between a filter and a matrix (the input). In other words, we have a filter flowing over an image to learn and map features. This makes the convolutional network particularly efficient because it leads to sparse interaction and fewer parameters to save.

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/cnn.gif?raw=true)

A convolutional network is the repetition of these elements, in which convolution and pooling layers are interspersed, and then at the end, we have a series of fully convolutional layers

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/cnn3.png?raw=true)
  
</details>

<details>
  <summary><b>What is Recurrent Neural Network (RNN)?</b></summary>
  
  An **RNN** is a subtype of neural network that specializes in sequential data (a sequence of data X with x ranging from 1 to time t). They are recursive because they perform the same task for each element in the sequence, and the output for an element is dependent on previous computations. In simpler terms, at each input of the sequence they perform a simple computation and do an update of the **hidden state** (or memory), this memory is then used for subsequent computations. So this computation can be seen with a kind of roll because for each input the output of the previous input is important:

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/rnn.webp?raw=true)

More formally, the hidden state $\(h_t\)$ of an RNN at time step $\(t\)$ is updated by:

$$\[h_t = f(W_h h_{t-1} + W_x x_t + b)\]$$

And the output at time step $\(t\)$ is given by:

$$\[y_t = g(W_y h_t + b_y)\]$$

where:
- $\(h_t\)$ is the hidden state at time step $\(t\)$,
- $\(h_{t-1}\)$ is the hidden state at the previous time step $\(t-1\)$,
- $\(x_t\)$ is the input at time step $\(t\)$,
- $\(W_h\)$, $\(W_x\)$, and $\(W_y\)$ are the weight matrices for the hidden state, input, and output, respectively,
- $\(b\)$ and $\(b_y\)$ are the bias terms,
- $\(f\)$ is the activation function for the hidden state, often tanh or ReLU,
- $\(g\)$ is the activation function for the output, which depends on the specific task (e.g., softmax for classification).

which can be also represented as:

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/description-block-rnn-ltr.png?raw=true)

RNNs can theoretically process inputs of indefinite length without the model increasing in size (the same neurons are reused). The model also takes into account historical information, and weights are shared for various interactions over time. In reality, they are computationally slow, inefficient to train, and after a few time steps forget past inputs.
  
</details>

<details>
  <summary><b>What is a deep network?</b></summary>

  **Deep neural networks** are basically neural networks in which there are many more layers. Today almost all of the most widely used models belong to this class. 

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/Artificial-Neural-Network-Vs-Deep-Neural-Network-14.png?raw=true)

Obviously, models with more hidden layers can build a more complex and sophisticated representation of the data. However, this comes at a cost in terms of data required (more layers, more data, especially to avoid overfitting), time, and computation resources (more parameters often take more time and more hardware). In addition, training requires special tunings to avoid problems such as overfitting, underfitting, or vanishing gradients.
  
</details>

<details>
  <summary><b>What’s the Dying ReLU problem? </b></summary>

The rectifier or **ReLU (rectified linear unit)** activation function has a number of advantages but also a number of disadvantages: 
* It is not zero differentiable (this is because it is not "smooth" at zero).
* Not zero-centered.
* Dying ReLU problem

The **Dying ReLU problem** refers to the scenario in which many ReLU neurons only output values of 0. In fact, if before the activation function the output of the neuron is less than zero, after ReLU is zero:

```math
\text{ReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
```

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/1024px-ReLU_and_GELU.svg.png?raw=true) *Plot of the ReLU rectifier (blue) and GELU (green) functions near x = 0, from Wikipedia*

The problem occurs when most of the inputs are negative. The worst case is if the entire network dies, at which point the gradient fails to flow during backpropagation and there will no longer be an update of the weights. The entire or a significant part of the network then becomes inactive and stops learning. Once this happens there is no way to reactivate it.

The Dying ReLU problem is related to two different factors:
* **High learning rate.** The high learning rate could cause the weight to become negative since a large amount will be subtracted, thus leading to negative input for ReLU.
* **Large negative bias**. Since bias contributes to the equation this is another cause.

So the solution is either to use a small learning rate or to test one of several alternatives to ReLU

 Suggested lecture:
  * [Dying ReLU and Initialization: Theory and Numerical Examples](https://arxiv.org/abs/1903.06733)

</details>

<details>
  <summary><b>What is the vanishing gradient? </b></summary>

The **Vanishing Gradient Problem** refers to the decreasing gradient and its approach to zero as different layers are added to a neural network. The deeper the network gets, the less gradient reaches the first few layers and the more difficult it becomes to train.

This is clearly understandable if the sigmoid function is used as the activation function. The sigmoid function squishes a large input between 0 and 1, so a large change in input does not correspond to a large change in output. In addition, its derivative is small for a large input X

**sigmoid activation function**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**derivative**:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/sigmoid_derivative.webp?raw=true) * from [here](https://isaacchanghau.github.io/img/deeplearning/activationfunction/sigmoid.png)*

 When there are few layers this is not a problem, but the more layers added the more it reduces the gradient and impacts training. Since backpropagation starts from the final layer to the initial layers, if there are n layers with sigmoid it means that n small derivatives are multiplied (backpropagation in fact uses the chain rule of derivatives). With little gradient, we will have little update of the first layers, so these layers will not be trained efficiently

The **ReLU** has been used as a solution, and in fact little by little it has become the most widely used function in neural networks. For particularly deep networks, **residual connections** have also been used, which precisely skips the activation function and thus avoids derivative reduction. Also, to reduce input space, **batch normalization** is another solution, thus preventing the input from adding the outer edges of the sigmoid
  
</details>

<details>
  <summary><b>What is the dropout? how I should use it efficiently?</b></summary>
  
**Dropout** is a regularization technique that aims to reduce network complexity to avoid overfitting. The problem is that models can learn statistical noise, the best way to avoid this is to change parameters, get different models, and aggregate. Obviously, this would be very computationally expensive. Dropout instead allows for the implicit ensemble.

*"Dropout is a technique that addresses both these issues. It prevents overfitting and
provides a way of approximately combining exponentially many different neural network
architectures efficiently. The term “dropout” refers to dropping out units (hidden and
visible) in a neural network. By dropping a unit out, we mean temporarily removing it from
the network, along with all its incoming and outgoing connections, as shown in Figure 1.
The choice of which units to drop is random. " -[source: original papers](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/dropout.png?raw=true) * from [the original papers](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*

During training a certain amount of neurons (decided with a probability p) is deactivated. If the probability p is 50 % for a layer, it means that randomly 50% of the neurons will be set to zero. This means that the model cannot rely on a particular neuron for training nor on the combination of neurons, but will learn different representations.  This acts on overfitting because during overfitting one neuron might compensate for the error of another neuron. This process is called **co-adaptations** in which several neurons are in "collusion" and reduces the generalization abilities of the neurons. If we use dropout instead, we prevent neuron co-adaptation because some neurons are set to zero in random manner.

*" According to this theory, the role of sexual reproduction is not just to allow useful new genes to spread throughout the population, but also to facilitate this process by reducing complex co-adaptations that would reduce the chance of a new gene improving the fitness of an individual. Similarly, each hidden unit in a neural network trained with dropout must learn to work with a randomly chosen sample of other units. This should make each hidden unit more robust and drive it towards creating useful features on its own without relying on other hidden units to correct its mistakes. However, the hidden units within a layer will still learn to do different things from each other. " -[source: original papers](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*

This also allows us to learn features that are better generalizable:

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/dropout2.png?raw=true) * from [the original papers](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*

During training, if you set the probability to 50 % the remaining neurons are rescaled by an equivalent factor (e.g. 2x). In inference, on the other hand, the probability p is zero and all neurons are active.

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/dropout1.png?raw=true) * from [the original papers](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*

 Tips for using Dropout: 
 * When using ReLU according to some it would be better, to put it before the activation function (fully connected, dropout, ReLu).
 * The general rule of thumb would be to use low dropout rates first (p= 0.1/0.2) and then increase until no decrease in performance. In the original article, they suggest 0.5 as a general value for a variety of tasks and for hidden units. In some articles, they suggest 0.8 for the input layer (this is how 20% of neurons are considered) and 50% for hidden layers. Or at any rate a higher p in the first few layers.
 * Dropout is especially recommended for large networks and small datasets

</details>

<details>
  <summary><b>What is the batch normalization? how I should use it efficiently?</b></summary>
!
</details>

<details>
  <summary><b>How to deal with overfitting in neural networks?</b></summary>
  
  *"The central challenge in machine learning is that we must perform well on new, previously unseen inputs — not just those on which our model was trained. The ability to perform well on previously unobserved inputs is called generalization."-[source](https://www.deeplearningbook.org/)* 


  Neural networks are sophisticated and complex models that can be composed of a great many parameters. This makes it possible for neural networks to store patterns and correlations spurious that are only present in the training set.  There are several techniques that can be used to reduce the risk of **overfitting**

**collecting more data and data augmentation**

Obviously, the best way is to collect more data, especially quality data. In fact, the model is exposed to more patterns and needs to identify relevant ones

Data augmentation simulates having more data and makes it harder for the model to learn spurious correlations or store patterns or even whole examples (deep networks are very capable in terms of parameters).

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/img_data_augmentation.jpg?raw=true)
*from [here](https://ai.google.dev/examples/clustering_with_embeddings)*

</details>
  
<details>
  <summary><b>What is the lottery ticket hypothesis?</b></summary>

The **lottery ticket hypothesis** was proposed in 2019 to explain why neural networks are pruned after training. In other words, once we have trained a neural network with lots of parameters we want to remove the weights that do not serve the task and create a lighter network (pruning). This allows for smaller, faster neural networks that consume fewer resources. Many researchers have wondered, but can't we eliminate the weights before training? If these weights are not useful afterward, they may not be useful during training either. 

*"The Lottery Ticket Hypothesis. A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.
"-[source](https://arxiv.org/abs/1803.03635)*


![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/lotteryticket.webp?raw=true)
*from [here](https://towardsdatascience.com/saga-of-the-lottery-ticket-hypothesis-af30091f5cb)*

 What the authors do is a process called **Iterative Magnitude Pruning**, basically, they start by training the network, eliminate all the smaller weights, and then extract a subnetwork. This subnetwork is initialized with small, random weights and they re-train until convergence. This subnetwork is called a "winning ticket" because randomly it received the right weights so that it could be the one with the best performance.

Now, this leads to two important considerations: There is a random subnetwork that is more computationally efficient and can be further trained to improve performance. Also, this subnetwork has better generalization capabilities. If it could be identified in a pre-training manner it would reduce the need to use large dense networks. 

According to some authors, the lottery ticket hypothesis is one of the reasons why neural networks form sophisticated circuits. Thus, it is not that weights gradually improve, but instead improve if these circuits are already present (weights that have won the lottery). This would then be the basis of grokking


 
 Suggested lecture:
  * [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
  
</details>


<details>
  <summary><b>What is grokking?</b></summary>
Overfitting is considered one of the major problems in neural networks and is defined when a model fails to generalize. An overfitting model shows good results on the training dataset, but poor results on test data (so it fails to generalize what it has learned in the training data). Overfitting can have many causes such as little data, lack of regularization, and model being too complex (see above).

Recently, however, the concept of overfitting has been meesso challenged by the so-called [Grokking](https://arxiv.org/abs/2201.02177). **Grokking** is defined as a delayed generalization, at the first stage the model seems to memorize the training set data and not learn generalization (this is seen from the loss and accuracy curves) at a certain time continuing training there is a rapid decrease in the validation loss (also known as “grok”). 

![grokking](https://github.com/SalvatoreRa/tutorial/blob/main/images/grokking.png?raw=true)
*Grokking: A dramatic example of generalization far after overfitting on an algorithmic dataset. from [here](https://arxiv.org/pdf/2201.02177)*

We discussed this in detail [in this article](https://levelup.gitconnected.com/grokking-learning-is-generalization-and-not-memorization-52c43c9025e4), but basically there are various forces at work: 
* The model tends to memorize elements because it is more efficient. These circuits memorize training examples.
* under regularization forces such as weighting decay, slowly generalization circuits emerge. These circuits are able to understand the patterns underlying the data.
* This balance is dependent on various elements such as dataset size.

Grokking seems more like a theoretical case without practical applications, especially since it needs many iterations to emerge. A [paper](https://arxiv.org/pdf/2405.20233) was recently presented that discusses the possibility of creating an algorithm called _Grokfast_, to accelerate model convergence toward generalization.

The system decomposes the gradient of a parameter into two components: a fast-varying component and a slow-varying component. The former is responsible for overfitting, and the latter is responsible for generating (inspired by circuits described in other articles). By exploiting this you can then speed up convergence, and simply strengthen the influence of the slow-varying component. [Here,](https://github.com/ironjr/grokfast) the code.

Articles describing in detail:
  * [Grokking: Learning Is Generalization and Not Memorization](https://levelup.gitconnected.com/grokking-learning-is-generalization-and-not-memorization-52c43c9025e4)

</details>


## Embeddings

<details>
  <summary><b>What is an embedding?</b></summary>
  
  An **embedding** is a low-dimensional space representation of high-dimensional space vectors. For example, one could represent the words of a sentence with a sparse vector (1-hot encoding or other techniques), and embedding allows us to obtain a compact representation. Although embedding originated for text, it can be applied to all kinds of data: for example, we can have a vector representing an image

  sparse word representation:

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/word_repr.png?raw=true)
*from [here](https://arxiv.org/pdf/2010.15036.pdf)*


In general, the term embedding became a fundamental concept of machine learning after 2013, thanks to **Word2Vec**. Word2Vec made it possible to learn a vector representation for each word in a vocabulary.  This vector captures features of a word such as the semantic relationship of the word, definitions, context, and so on. In addition, this vector is numeric and can be used for operations (or for downstream tasks such as classification)

  ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/word2vec.png?raw=true)
*from [here](https://arxiv.org/pdf/2010.15036.pdf)*

Word2Vec then succeeds in grouping similar word vectors (distances in the embedding space are meaningful). Word2Vec estimates the meaning of a word based on its occurrences in the text. The system is simple, we try to predict a word by its neighbors or its context. In this way, the model learns the context of a word

 Suggested lecture:
 * [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

</details>


<details>
  <summary><b>What are embedding vectors and latent space? What is a latent representation?</b></summary>

As mentioned above an **embedding vector** is the representation of high dimensional data in vectors that have a reduced size. In general, they are continuous vectors of real numbers, although the starting vectors may be sparse (0-1) or of integers (pixel value). The value of these embedding vectors is that they maintain a concept of similarity or distance. Therefore, two elements that are close in the initial space must also be close in the embedding space.

The **representation** means a transformation of the data. For example, a neural network at each layer learns a representation of the data. intermediate, i.e., layer representations within the neural network are also called latent representation (or latent space). The representation can also be the output of a neural network, and the resulting vectors can also be used as embeddings. 

For example, we can take images, pass them through a transformer, and use the vectors obtained after the last layer. These vectors have a small size and are both a representation and embedding. So often the terms have the same meaning. Since a text like hot-encoding is a representation, in some cases the terms differ
  
</details>

<details>
  <summary><b>Can we visualize embedding? It is interesting to do it?</b></summary>
  
  As mentioned, _distances are meaningful in the embedding space_. In addition, operations such as searching for similar terms can be performed. These embeddings have been used to search for similar documents or conduct clustering. 

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/vect.png?raw=true)
*from [here](https://developers.google.com/machine-learning/crash-course/embeddings/translating-to-a-lower-dimensional-space)*

After getting an embedding it is good practice to conduct **a visual analysis**, it allows us to get some visual information and understand if the process went well. Because embeddings have many dimensions (usually up to 1024, although they can be more), we need to use techniques to reduce the dimensions such as PCA and t-SNE to obtain a two-dimensional projection

 ![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/tsne_vect.png?raw=true)
*from [here](https://ai.google.dev/examples/clustering_with_embeddings)*
  
</details>

## Transformer

<details>
  <summary><b>What is self-attention?</b></summary>
  !
</details>

<details>
  <summary><b>What is a transformer?</b></summary>
  !
</details>

<details>
  <summary><b>What is a Vision Transformer (ViT)?</b></summary>
  !
</details>


<details>
  <summary><b>Why there are so many transformers?</b></summary>
  !
</details>

<details>
  <summary><b>What is a mamba model? Is it an alternative to a transformer?</b></summary>

  In recent times there has been talk of Mamba and **State Space Models (SSMs)** as possible alternatives to transformers. Recently these models seem promising because they have a nearly linear computational cost that is especially attractive when long sequences of text (up to one million tokens) have to be modeled.

 ![mamba scaling](https://github.com/SalvatoreRa/tutorial/blob/main/images/mamba_scaling.png?raw=true)
*from [here](https://arxiv.org/abs/2312.00752)*

The authors of the models describe it as *" Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba  achieves state-of-the-art performance across several modalities such as language, audio, and genomics"*

SSMs originated as a framework for describing the behavior of a system dynamically over time. In a simple way, considering a maze, the “state space” is the map of all possible locations (or states), and the “state space representation” can be considered the map. This map tells us where we can go, how to go there, and where we are at that moment. Where we are and how far we are from the exit of the maze could be described with a vector (“state vectors”). In the context of a neural network for text, given the state of the system (or hidden state) we can generate a new token (or metaphorically switch state or move in the map). 

The basic assumption is that if one knows the current state of the world and how it will evolve, one can decide how to move.

More technically, SSMs then try for an input sequence x(t) to map it to a latent state representation h(t) and predict an output y(t). For example, given a text sequence, generate a latent representation and use it to predict the next word. A dynamical system can be predicted from its state at each time t using these two equations:

equation 1: h'(t) = A h(t) + B x(t),

equation 2: y(t) = C h(t) + D x(t),

The goal is then to determine h(t) so given x(t) we can then determine the output y(t). Equation 1 tells us that the state h changes as a function of the current state (based on a matrix A) and the input x (and a matrix B). Equation 2 tells us that the output is obtained grazio the state h (and a matrix C) and the input x (and an associated matrix D). In simple words, given an input x the state h is changed, and the output is affected by both the input x and the state h (for an input x we have the update of the state h and an output y). More formally, the evolution of the system depends on newly acquired information and its current state. Applied to a language model, arriving at an input x with a model with a state h we can predict the next token y (if you notice it is very close to an autoregressive language model or how transformers work).

 ![mamba structure](https://github.com/SalvatoreRa/tutorial/blob/main/images/mamba_structure.png?raw=true)
*from [here](https://arxiv.org/abs/2312.00752)*

Briefly, we can consider the different matrixes as:
* A is the transition state matrix, which is guiding the transition from one state to another. Intuitively it represents how we can forget the least relevant part of the state. 
* B maps the input to the new state, controlling which part of the input we need to remember. 
* C allows us to map the output from the model state, or how to use the model state to make a prediction.
* D is considered a kind of skip connection, or how the input affects the prediction

Calculating the state representation h(t) analytically is complex, especially if the signal is continuous. Since text is a discrete input by nature, discretizing the model makes our lives easier. **Zero-Order Hold (ZOH)** is the technique that is used in Mamba to transform the model. After applying the ZOH, the equations are:

$$h_k = \overline{A} h_{k-1} + \overline{B} x_k$$

$$y_k = C h_k$$

where $$\overline{A} = \exp(\Delta A)$$, and $$\overline{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B$$, $$k$$ is the discrete time step.

 ![mamba structure unrolled](https://github.com/SalvatoreRa/tutorial/blob/main/images/mamba_unrolled.png?raw=true)
*from [here](https://arxiv.org/pdf/2408.01129)*

In addition, we can also see this process as a convolution, in which we apply kernel sliding on the various tokens. At each time step, we can calculate the output in this way:

 ![mamba convolutional calculation](https://github.com/SalvatoreRa/tutorial/blob/main/images/convolutional_form.png?raw=true)
*from [here](https://arxiv.org/pdf/2408.01129)*

The three representations we have seen (continuous representation and its discritization into recurrent and convolutional) have different advantages and disadvantages. Recurrent representation is efficient in inference but does not allow parallel training. So training is conducted with convolutional representation (it allows parallelization of training). 

Another interesting modification in Mamba is the use of High-order Polynomial Projection Operators (HiPPO) to initialize the matrix A during training. This is used to compress the input signals into vectors of coefficients. The idea is to exploit d this concept in the A-matrix, so that this captures the recent tokens and there is a decay of information for the old tokens. After all, the A-matrix is used to capture the information from previous states to produce the new state. In this way we improve the ability of the model in handling long-range dependencies.

In addition, Mamba uses two particular additions: 
* **selective scan algorithm** to filter out irrelevant information. This is especially important to allow the model to be context-aware and not treat all tokens equally.
* **hardware-aware algorithm** that allows it to store intermediate results. Hardware-aware algorithm allows it to better exploit the capabilities of the GPU (similar to what flash attention does).

An SSM model compresses the entire history (i.e., everything seen up to that model) and does so efficiently. Transformers do not compress the story, they are, however, very powerful to look at and attend to the sequence (thus searching for what is important and modeling relationships). In a sense, the internal state of a transformer can be seen almost as the cache of the whole hystory. SSMs are not as powerful, so Mamba tries to have a state that is both efficient and powerful. Therefore, the B and C matrices have a dynamic size that changes in dependence of the input and are different for each input token (thus ensuring context awareness). These two matrices thus help to choose which information to retain and which not to. 

These improvements can then be seen in the Mamba block. Here the SSM conducts discretization, HiPPO initialization, Selective scan algorithm and is accelerated thanks to hardware-aware algorithm

 ![mamba block](https://github.com/SalvatoreRa/tutorial/blob/main/images/mamba_block.png?raw=true)
*from [here](https://arxiv.org/pdf/2408.01129)*


</details>


## Large Language Models

<details>
  <summary><b>What is a Large Language Model (LLM)?</b></summary>
  
  *"language modeling (LM) aims to model the generative likelihood of word sequences, so as to predict the probabilities of future (or missing) tokens"* -from [here](https://arxiv.org/abs/2303.18223)

  In short, starting from a sequence x of tokens (sub-words) I want to predict the probability of what the next token x+1 will be.  An LLM is a model that has been trained with this goal in mind. Large because it is a model that has more than 10 billion parameters (by convention).

These LLMs are obtained by scaling from the transformer and have general capabilities. Scaling means increasing parameters, training budget, and training dataset.

*"Typically, large language models (LLMs) refer to Transformer language models that contain hundreds of billions (or more) of parameters, which are trained on massive text data"* -from [here](https://arxiv.org/abs/2303.18223)

![LLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/LLM.png?raw=true)
*from the [original article](https://arxiv.org/abs/2303.18223)*

LLMs have been able to be flexible for so many different tasks, have shown reasoning skills, and all this just by having text as input. That's why so many have been developed in recent years: 
* **closed source** such as ChatGPT, Gemini, or Claude.
* **Open source** such as LLaMA, Mistral, and so on.

![LLM](https://github.com/SalvatoreRa/tutorial/blob/main/images/LLM2.png?raw=true)
*from the [original article](https://arxiv.org/abs/2303.18223)*

Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [The Infinite Babel Library of LLMs](https://towardsdatascience.com/the-infinite-babel-library-of-llms-90e203b2f6b0)
 
  Suggested lecture:
  * [Tabula Rasa: Large Language Models for Tabular Data](https://levelup.gitconnected.com/tabula-rasa-large-language-models-for-tabular-data-e1fd781946fa)
  * [Speak Only About What You Have Read: Can LLMs Generalize Beyond Their Pretraining Data?](https://pub.towardsai.net/speak-only-about-what-you-have-read-can-llms-generalize-beyond-their-pretraining-data-041704e96cd5)
  * [Welcome Back 80s: Transformers Could Be Blown Away by Convolution](https://levelup.gitconnected.com/welcome-back-80s-transformers-could-be-blown-away-by-convolution-21ff15f6d1cc)
  * [a good survey on the topic](https://arxiv.org/abs/2303.18223)
  * [another good survey on the topic](https://arxiv.org/abs/2402.06196)

</details>

<details>
  <summary><b>What does it mean emergent properties? what it is the scaling law?</b></summary>

OpenAI proposed in 2020 a _power law for the performance of LLMs_: according to this scaling law, there is a relationship with three main factors: y model size (N), dataset size (D), and the amount of training compute (C). Given these factors we can derive the performance of the models:

![scaling law](https://github.com/SalvatoreRa/tutorial/blob/main/images/scaling_law.png?raw=true)
*from the [original article](https://arxiv.org/abs/2001.08361)*

A [later work](https://arxiv.org/abs/2407.13623) suggests that vocabulary size also follows a scaling law (model performance is also impacted by vocabulary size). A larger vocabulary allows more concepts and nuances to be represented. According to the authors, vocabulary size in today's models is not optimal but is underestimated.

![Vocabulary scaling law](https://github.com/SalvatoreRa/tutorial/blob/main/images/vocabulary_scaling_law.png?raw=true)
*from the [original article](https://arxiv.org/abs/2407.13623)*

**Emergent properties** are properties that appear only with scale (as the number of parameters increases)

*"In the literature, emergent abilities of LLMs are formally defined as “the abilities that
are not present in small models but arise in large models”, which is one of the most prominent features that distinguish LLMs from previous PLMs."*-[source](https://arxiv.org/pdf/2303.18223.pdf)

![emergent_properties](https://github.com/SalvatoreRa/tutorial/blob/main/images/emergent_properties.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2206.07682.pdf)*

![emergent_properties](https://github.com/SalvatoreRa/tutorial/blob/main/images/emergent_properties2.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2206.07682.pdf)*

On the other hand, not everyone agrees on the real existence of these emerging properties

*" There are also extensive debates on the rationality of emergent abilities. A popular speculation is that emergent abilities might be partially attributed to the evaluation setting for special tasks (e.g., the discontinuous evaluation metrics)."*-[source](https://arxiv.org/pdf/2303.18223.pdf)

Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [Emergent Abilities in AI: Are We Chasing a Myth?](https://towardsdatascience.com/emergent-abilities-in-ai-are-we-chasing-a-myth-fead754a1bf9)

 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [About emergent properties](https://arxiv.org/pdf/2206.07682.pdf)
  * [a good survey on LLMs, scaling law, and so on](https://arxiv.org/abs/2303.18223)

  
</details>

<details>
  <summary><b>What does it mean context length?</b></summary>
  
  **Context length** is the maximum amount of information an LLM can take as input. It is generally measured in tokens or subwords. So an LLM with a context length of 1000 tokens can take about 750 words (as a rule of thumb, a token is considered to be 3/4 of a word). Context length has an effect on accuracy, consistency, and how much information it can parse: 
  
* The more context length the slower the model generally is. 
* Models with small context lengths use resources more efficiently. 
* Larger context lengths have more conversation memory and more contextual understanding.

There are methods to extend the context length of the models:

![context_length](https://github.com/SalvatoreRa/tutorial/blob/main/images/context_length.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2402.02244.pdf)*

Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [Speak to me: How many words a model is reading](https://towardsdatascience.com/speak-to-me-how-many-words-a-model-is-reading-331e3af86d27)

 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [About extending context length](https://arxiv.org/abs/2402.02244)
  * [About extending context length](https://arxiv.org/abs/2401.07872)
  
    
</details>


<details>
  <summary><b>What does it mean LLM's hallucination?</b></summary>
Anyone who has interacted with ChatGPT will notice that the model generates responses that seem consistent and convincing but occasionally are completely wrong. 

![hallucination](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination.png?raw=true)
*from the [original article](https://arxiv.org/abs/2311.05232)*

Several solutions have obviously been proposed: 
* Provide some context in the prompt (an article, Wikipedia, and so on). Or see RAG below.
* If you have control over model parameters, you can play with temperature or other parameters.
* Provide instructions to the model to answer "I do not know" when it does not know the answer.
* Provide examples so it can better understand the task (for reasoning tasks).
* Adding an external memory

Investigating further, [this study](https://arxiv.org/pdf/2304.13734) provides three reasons why hallucinations emerge. Specifically discussing a model that generates a sequence of tokens in an autoregressive manner: 
* The LLM commits to a token at a time. Even when we choose a low temperature, during decoding we maximize the likelihood of each token given the previous tokens, but the probability of the entire correct sequence may be low. For example, once part of the sequence is generated the model will continue from there and will not correct (For a sequence to be completed "Pluto is the" if the model continues as "Pluto is the smallest," it will likely continue with "Pluto is the smallest dwarf planet in our solar system." and not with the correct completion).  The model completes a sentence that it does not know how to complete. For example, for the description of a city, the model predicts to describe its population, but having no information about its population in its parametric memory generates a hallucination.
* The second reason is that although there are multiple correct chances to complete the sequence, the incorrect one has a higher likelihood.
* Third, when we use an LLM we do not use the maximal probability for the next word, but we sample according to the distribution over the words. This makes it so that in some cases we sample words that result in false
information.

![hallucination LLM causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination_causes.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2304.13734)*

Other studies indicate among the causes as problems with the training dataset. Lack of relevant data to answer a question can lead to the emergence of hallucinations. The presence of duplicated data. In fact, duplicated data impacts the performance of the model.  With smaller models seeming more sensitive to repeated data. Repetitions seem to cause the model to store this data and lead to performance degradation (for more details you can read [here](https://arxiv.org/abs/2205.10487) and [here](https://aclanthology.org/2022.naacl-main.387/)). For some authors, hallucinations are also derived from inherent model limitations. In [this study](https://arxiv.org/abs/2305.14552) show that LLMs still rely on memorization at the sentence level and statistical patterns at the corpora level instead of robust reasoning, this is one of the reasons for the manifestation of hallucinations. This is also observed by the fact that LLMs are sensitive to reverse curse (lack of logical deduction, where an LLM trained on A implies B, hallucinates when questioned on B implies A, [here more details](https://arxiv.org/abs/2305.14552)). Other causes of hallucinations have been defined as the tendency of the model to be overconfident ([here](https://arxiv.org/abs/2307.11019)), favoring co-occurrences words over factual answers and thus generating spurious correlations ([here](https://arxiv.org/abs/2310.08256)) and the tendency to sycophancy to please the user ([here](https://arxiv.org/abs/2308.03958)). 

![an example of sycophancy](https://github.com/SalvatoreRa/tutorial/blob/main/images/sycophancy.png?raw=true)
*an example of sycophancy, from the [original article](https://arxiv.org/pdf/2308.03958)*

Recent studies show how fine-tuning and instruction tuning can increase an LLM's tendency to hallucinate (there is a correlation between examples unknown to the model and the tendency to hallucinate on prescient knowledge). Hallucinations then may also emerge due to a discrepancy between new knowledge and previously acquired knowledge (more details [here](https://arxiv.org/abs/2405.05904))

There is a dissonance between what is the meaning of the term "hallucination" in human psychology ("when you hear, see, smell, taste, or feel things that appear to be real but only exist in your mind") and what is meant by hallucination in machine learning. In a [recent article](https://arxiv.org/abs/2402.01769) they took care to align these two definitions. They divided hallucinations that are seen in the case of LLM into different types. This new classification is interesting because it is difficult to be able to resolve all causes of hallucinations in LLM with one approach. Instead, by having a classification of the various subtypes, one can think about acting on each subtype (which is perhaps the one most relevant to our task):

_By grounding our discussion in specific psychological constructs, we seek to shed light on these phenomena in
language models, paving the way for the development of targeted solutions for different types of ”hallucinations.” - [source](https://arxiv.org/abs/2402.01769)_

![hallucination LLM causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/llm_hallucination_psicology.png?raw=true)
*from the [original article](https://arxiv.org/abs/2402.01769)*

For example, **confabulation** is a hallucination that emerges from the LLM unpredictably, owing to internal factors that are unrelated to the prompt. In a sense, this type of hallucination is associated with the LLM's uncertainty in responding to the prompt. In [this paper](https://www.nature.com/articles/s41586-024-07421-0) they show that a high uncertainty in the response is an indication of confabulation (this uncertainty can be estimated with an entropy that is associated with the meaning of the response).

Another type of hallucination is **contextual hallucination**. In this case, although we provide the context (and thus the correct facts) in the prompt the model fails to generate the correct output. According to this [study]() contextual hallucinations are related to the extent to which an LLM attends to the provided contextual information. In other words, it depends on the relationship between the attention (attention weights) associated with the context and the attention devoted to the newly generated tokens. Therefore, one can classify when a model will generate these kinds of hallucinations by extracting the attention weights and constructing a linear classifier.

![hallucination RAG causes](https://github.com/SalvatoreRa/tutorial/blob/main/images/hallucination_contextual.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2304.13734)*

Articles describing in detail:
  * [A Requiem for the Transformer?](https://towardsdatascience.com/a-requiem-for-the-transformer-297e6f14e189)
  * [AI Hallucinations: Can Memory Hold the Answer?](https://towardsdatascience.com/ai-hallucinations-can-memory-hold-the-answer-5d19fd157356)
  * [Chat Quijote and the Windmills: Navigating AI Hallucinations on the Path to Accuracy](https://levelup.gitconnected.com/chat-quijote-and-the-windmills-navigating-ai-hallucinations-on-the-path-to-accuracy-0aaecf46354c)
 
  Suggested lecture:
  * [Speak Only About What You Have Read: Can LLMs Generalize Beyond Their Pretraining Data?](https://pub.towardsai.net/speak-only-about-what-you-have-read-can-llms-generalize-beyond-their-pretraining-data-041704e96cd5)
  * [a good survey on the topic](https://arxiv.org/abs/2311.05232)
  * [discussing hallucination in psychological terms](https://arxiv.org/abs/2402.01769)


  
</details>

<details>
  <summary><b>Do LLMs have biases?</b></summary>
  
  Yes. 

  Garbage in, garbage out. Most of the biases come from the training dataset and LLMs inherit them. Biases are a particularly tricky problem especially if the model is to be used for sensitive applications

  *" Laying behind these successes, however, is the potential to perpetuate harm. Typically trained on an enormous scale of uncurated Internet-based data, LLMs inherit stereotypes, misrepresentations, derogatory and exclusionary language, and other denigrating behaviors that disproportionately affect already-vulnerable and marginalized communities"*-[source](https://arxiv.org/pdf/2309.00770.pdf)

  ![bias](https://github.com/SalvatoreRa/tutorial/blob/main/images/bias.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2309.00770.pdf)*

  Articles describing in detail:
  * [PrAIde and Prejudice: Tracking and Minimize Political Bias in LLMs](https://levelup.gitconnected.com/praide-and-prejudice-tracking-and-minimize-political-bias-in-llms-47f82d354514)

 Suggested lecture:
  * [Bias and Fairness in Large Language Models: A Survey](https://arxiv.org/abs/2309.00770)

</details>

<details>
  <summary><b>What are adversarial prompts?</b></summary>

  **Adversarial prompting** is an interesting field of both research and applications because it serves to understand the limits and safety of a model. These techniques have been shown to work on ChatGPT and other 

For example, prompt injection is a technique in which you insert several prompts one safe and one that is used to obtain unexpected behavior instead.

an example of prompt injection from Twitter:

![prompt](https://github.com/SalvatoreRa/tutorial/blob/main/images/prompt_injection.png?raw=true)

Jailbreak and DAN (Do anything now) are examples of techniques used to overcome safety controls and try to make a model say something illegal

![jailbreak](https://github.com/SalvatoreRa/tutorial/blob/main/images/jailbreak.png?raw=true)

Articles discussing the topic:
  * [The AI worm and the LLM leaf](https://levelup.gitconnected.com/praide-and-prejudice-tracking-and-minimize-political-bias-in-llms-47f82d354514)

 Suggested lecture:
  * [The Waluigi Effect (mega-post)](https://levelup.gitconnected.com/the-ai-worm-and-the-llm-leaf-6132d60c11be)
  
</details>

## Prompt engineering

<details>
  <summary><b>What is a prompt? What is prompt engineering?</b></summary>
  The prompt is a textual instruction used to interact with a Large Language Model. Prompt     engineering, on the other hand, is a set of methods and techniques for developing and optimizing prompts. prompt engineering is specifically designed to improve the capabilities of models for complex tasks that require reasoning (question answering, solving mathematical problems, and so on).


  Prompt engineering can have other functions such as improving LLM safety (instructions that serve to prevent the model from responding in a toxic manner) or providing additional knowledge to a model


  Prompt engineering can have other functions such as improving LLM safety (instructions that serve to prevent the model from responding in a toxic manner) or providing additional knowledge to a model


  A prompt, in its simplest form, is a set of instructions or a question. In addition, it might also contain other elements such as context, inputs, or examples.


  prompt:

  ```
  Which is the color of the sea?
  ```
output from a LLM:
  ```
  Blue
  ```

Formally, a prompt contains or more of these elements: 
* **Instruction**. Information about the task you want the LLM execute
* **Context**. Additional or external information the model has to take into account.
* **Input data**. Input data that can be processed
* **Output indicator**. We can provide additional requirements (type or format of the output)

  prompt:

  ```
    Classify the sentiment of this review as in the examples:

  The food is amazing - positive
  the chicken was too raw - negative
  the waitress was rude - negative
  the salad was too small -
  ```

  Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)
 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)

</details>

<details>
  <summary><b>What is zero-shot prompting? What is few-shot prompting?</b></summary>

  LLMs are trained with a large amount of text. This allows them to learn how to perform different tasks. These skills are honed during instruction tuning. As shown [in this article](https://arxiv.org/abs/2109.01652), instruction tuning improves the model's performance in following instructions. During reinforcement learning from human feedback (RLHF) the model is aligned to follow instructions.


Zero-shot means that we provide nothing but instruction. Therefore, the model must understand the instructions and to execute it:

 zero-shot prompt:

  ```
    Classify the sentiment of this review :
    review: the salad was too small
    sentiment:
  ```

This is not always enough, so sometimes it is better to provide help to the model to understand the task. In this case, we provide some examples, that help the model improve its performance

This was discussed during the GPT-3 presentation:

![few shot learning](https://github.com/SalvatoreRa/tutorial/blob/main/images/few_shot%20learning.png?raw=true)
*from the [original article](https://arxiv.org/abs/2203.11171)*

Then a few-shot prompt is:

  ```
    Classify the sentiment of this review as in the examples:

  The food is amazing - positive
  the chicken was too raw - negative
  the waitress was rude - negative
  the salad was too small -
  ```

A couple of notes:
* Few-shot learning is one of the so-called emerging properties (because it would emerge with the scale).
* It is sensible to the format and used labels. However, new models are more robust to variations
* It is not optimal for complex tasks like mathematical reasoning or that require more reasoning steps



  Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)
 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [Emergent Abilities in AI: Are We Chasing a Myth?](https://towardsdatascience.com/emergent-abilities-in-ai-are-we-chasing-a-myth-fead754a1bf9)
</details>

<details>
  <summary><b>What is Chain-of-Thought (CoT)?</b></summary>


**Chain-of-thought (CoT)**

Chain-of-thought (CoT) Prompting is a technique that pushes the model to reason by intermediate steps. In other words, we are providing the model with intermediate steps to solve a problem so that the model understands how to approach a problem:

*"We explore the ability of language models to perform few-shot prompting for reasoning tasks, given a prompt that consists of triples: input, a chain of thought, and output. A chain of thought is a series of intermediate natural language reasoning steps that lead to the final output, and we refer to this approach as chain-of-thought prompting."--[source](https://arxiv.org/abs/2201.11903)*

  ![Cot Prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/cot.png?raw=true)
*from the [original article](https://arxiv.org/abs/2201.11903)*


**zero-shot Chain-of-thought (CoT)**

Instead of having to provide context, the authors of this study found that simply providing "Let's think step by step" was enough to suggest that the model reasons by intermediate steps:

*" Despite the simplicity, our Zero-shot-CoT successfully generates a plausible reasoning path in a zero-shot manner and reaches the correct answer in a problem where the standard zero-shot approach fails. Importantly, our Zero-shot-CoT is versatile and task-agnostic, unlike most prior task-specific prompt engineering in the forms of examples (few-shot) or templates (zero-shot)"--[source](https://arxiv.org/abs/2205.11916)*

![zero-shot Cot Prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/zeroshot-cot.png?raw=true)
*from the [original article](https://arxiv.org/abs/2205.11916)*

**Constrained-CoT**

is another variation in which you force the model to reduce the number of tokens in the output. According to the authors, today's LLMs are unnecessarily verbose and produce more tokens than necessary. In addition, larger models are generally more verbose and thus produce more tokens. This has a latency cost (as well as obviously computational) that can be problematic per service for users. Also, more verbiage means a lot of irrelevant unnecessary detail and a greater risk of hallucinations. In addition, CoT by requiring reasoning intermediates to be generated increases the number of tokens generated.

![verbosity of CoT prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/verbosity_cot.png?raw=true)
*CoT is increasing the number of generated tokens from  an LLM. from the [original article](https://arxiv.org/pdf/2407.19825)*

Therefore, the authors of [this study](https://arxiv.org/abs/2407.19825) suggest a new prompt that is a variation of the zero-shot prompt: *let's think step by step "and limit the length of the answer to n words* with n being the desired number of words.

![constrained CoT prompt](https://github.com/SalvatoreRa/tutorial/blob/main/images/constrained_cot.png?raw=true)
*Example of constrained CoT prompting, where n is 45. from the [original article](https://arxiv.org/pdf/2407.19825)*

For the authors, this prompt not only reduces the number of tokens generated but in several cases also leads to better reasoning (more exact answers on a reasoning database). This better reasoning is seen only with some models (LLaMA-2 70B and not with smaller models)


  Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)
  * [Multimodal Chain of Thoughts: Solving Problems in a Multimodal World](https://towardsdatascience.com/multimodal-chain-of-thoughts-solving-problems-in-a-multimodal-world-961a8ab9d0fa)
 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [original article CoT](https://arxiv.org/abs/2201.11903)
  * [original article zero-shot CoT](https://arxiv.org/abs/2205.11916)
  
</details>

<details>
  <summary><b>What is Self-Consistency?</b></summary>

Self-consistency is in simple terms getting more CoT and selecting the most consistent response:

*"We first prompt the language model with chain-of-thought prompting, then instead of greedily decoding the optimal reasoning path, we propose a “sample-and-marginalize” decoding procedure: we first sample from the language model’s
decoder to generate a diverse set of reasoning paths; each reasoning path might lead to a different final answer, so we determine the optimal answer by marginalizing out the sampled reasoning paths to find the most consistent answer in the final answer set."--[source](https://arxiv.org/abs/2203.11171)*

![Self-Consistency Prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/SELF-CONSISTENCY.png?raw=true)
*from the [original article](https://arxiv.org/abs/2205.11916)*

For the authors, this approach is analogous to how humans reason. In other words, humans test multiple ways of thinking and evaluate whether these lead to the same response. If there is consensus among the various responses, we humans feel more confident.

An example of question:

  ```
    Henry made two stops during his 60-mile bike trip. He first stopped after 20 miles. His second stop was 15 miles before the end of the trip. How many miles did he travel between his first and second stops?
  ```

Firs answer is wrong:

  ```
    He traveled 60 miles in total. He stopped after 20 miles, so he traveled 60 - 20 = 40 miles between the first and second stops. The answer is 40
  ```

second time:

  ```
   Henry traveled a total of 60 miles. His first stop was 20 miles in, and his second stop was 15 miles before the end. So between his first and second stops he travelled 60 - 20 -15 = 25 miles. The answer is 25. 
  ```

Third time:

  ```
    He made two stops during a 60-mile trip. The first was 20 miles into the trip. The second was 15 miles before the end of the trip. This means the second stop was 60 - 15 = 45 miles into the trip. Since he made the stops in order, the second stop must have been 45 - 20 = 25 miles after the first stop. The answer is 25.
  ```
We have an answer that is the majority. This a simplified version


  Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)

 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [original article](https://arxiv.org/abs/2203.11171)
 
  
</details>

<details>
  <summary><b>What is Tree of Thoughts (ToT)?</b></summary>

Tree of Thoughts (ToT) can be seen as a generalization of CoT, especially for more complex tasks. The idea is that the model maintains some degree of exploration to arrive at the solution. 

In this framework, the model produces several steps of thought as it moves toward solving the problem (similar to CoT or Self-consistency). In this case, though, the model not only generates these thoughts but evaluates them. In addition, there is a search algorithm (breadth-first search and depth-first search) that allows exploration but also backtracking

![TOT](https://github.com/SalvatoreRa/tutorial/blob/main/images/TOT.png?raw=true)
*from the [original article](https://arxiv.org/abs/2305.10601)*

An interesting development is to add an agent to conduct the search. In this article, they add a controller that precisely controls the search in the ToT

![TOT2](https://github.com/SalvatoreRa/tutorial/blob/main/images/TOT3.png?raw=true)
*from the [original article](https://arxiv.org/abs/2305.08291)*

Clarification these two methods are both complex and laborious. In one you have to generate the various steps, evaluate them, and conduct the research. In the second we have an external module trained with reinforcement learning that conducts the control and search. So instead of conducting multiple calls, [Hubert](https://github.com/dave1010/tree-of-thought-prompting) proposed a simple prompt to conduct ToT. This approach is simplified but seems to give superior results to simple CoT

```
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking,
then share it with the group.
Then all experts will go on to the next step, etc.
If any expert realises they're wrong at any point then they leave.
The question is...
```


Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)

 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [original article 1](https://arxiv.org/abs/2305.10601)
  * [original article 2](https://arxiv.org/abs/2305.08291)
</details>


<details>
  <summary><b>What is Emotion prompt?</b></summary>
  A new approach incorporating emotional cues improves LLM performance. The approach was inspired by psychology and cognitive science

  *"Emotional intelligence denotes the capacity to adeptly interpret and manage emotion-infused information, subsequently harnessing it to steer cognitive tasks, ranging from problem-solving to behaviors regulations. Other studies show that emotion regulation can influence human’s problem-solving performance as indicated by self-monitoring , Social Cognitive theory, and the role of positive emotions. "--[source](https://arxiv.org/abs/2307.11760)*

![EmotionPrompt](https://github.com/SalvatoreRa/tutorial/blob/main/images/EmotionPrompt.png?raw=true)
*from the [original article](https://arxiv.org/abs/2307.11760)*

The idea is to add phrases that can uplift the LLM ("you can do this," "I know you'll do great!") at the end of the prompt.

The advantages of this approach are: 
* improved performance.
* actively contribute to the gradients in LLMs by gaining larger weights, thus increasing the value of the prompt representation.
* Combining emotional prompts brings additional performance boosts.
* The method is very simple.

![EmotionPrompt](https://github.com/SalvatoreRa/tutorial/blob/main/images/EmotionPrompt2.png?raw=true)
*from the [original article](https://arxiv.org/abs/2307.11760)*


Articles describing in detail:
  * [Prompt Engineering to Leverage In-Context Learning in Large Language Models](https://pub.towardsai.net/prompt-engineering-to-leverage-in-context-learning-in-large-language-models-72296e1f09c3)

 
  Suggested lecture:
  * [All You Need to Know about In-Context Learning](https://towardsdatascience.com/all-you-need-to-know-about-in-context-learning-55bde1180610)
  * [original article](https://arxiv.org/abs/2307.11760)
   
</details>

<details>
  <summary><b>does it exist multimodal prompt engineering?</b></summary>

*Imagine reading a textbook with no figures or tables. Our ability to knowledge acquisition is greatly strengthened by jointly modeling diverse data modalities, such as vision, language, and audio. -[source](https://arxiv.org/abs/2302.00923)*

With the arrival of LLMs, there has been an increased interest in multimodal models. There are already several models that are multimodal, but most prompt engineering techniques are dedicated to traditional LLMs that do not take into account other modalities besides text.

![multimodal versus unimodal COT](https://github.com/SalvatoreRa/tutorial/blob/main/images/multimodal_vs_unimodal_cot.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

Chain-of-thoughts (CoT) is a technique to improve reasoning skills but is not adapted to the presence of other modalities. In [this article](https://arxiv.org/pdf/2302.00923) they proposed a variant definite **Multimodal-CoT**. In the same manner, multimodal-CoT decomposes multi-step problems into intermediate reasoning steps (rationale) and then infers the answer. They used in this work a model that is 1B and purpose-built to consider both image and textual modalities. This approach is a two-stage framework, in the first step the rationale (the chain of thoughts) is created based on the multimodal information and then after that, the model generates the answer.

![multimodal COT](https://github.com/SalvatoreRa/tutorial/blob/main/images/multimodal_cot.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

Here, is the pipeline in detail:

![multimodal COT](https://github.com/SalvatoreRa/tutorial/blob/main/images/multimodal_cot.png?raw=true)
*from the [original article](https://arxiv.org/pdf/2302.00923)*

Articles describing in detail:
  * [Multimodal Chain of Thoughts: Solving Problems in a Multimodal World](https://towardsdatascience.com/multimodal-chain-of-thoughts-solving-problems-in-a-multimodal-world-961a8ab9d0fa)

</details>

<details>
  <summary><b>What is ReAct Prompting?</b></summary>

**ReAct prompting** was introduced by [this article](https://arxiv.org/abs/2210.03629). It is based on the idea that humans can accomplish tasks and conduct reasoning about these tasks at the same time. Thus, in ReAct prompting, both reasoning and task actions are conducted. 

The process then alternates between retrieving information (which can come from external sources such as a search engine), evaluating the process, and if necessary conducting a plan update. This can then be combined with a chain of thought to follow a plan and track reasoning intermediates. This approach has shown promise especially when the model needs to conduct searches or take action.

*However, this “chain-of-thought” reasoning is a static black box, in that the model uses its own internal representations to generate thoughts and is not grounded in the external world, which limits its ability to reason reactively or update its knowledge - [source](https://arxiv.org/pdf/2210.03629)*

ReAct tries to solve this by giving the model access to external information. To avoid hallucinations or arriving at wrong conclusions, this approach tries to combine both the information that is received and an internal assessment. The approach combines reasoning traces and actions; it is also dynamic because it creates, maintains, and adjusts the plan as it unfolds

![ReAct prompting](https://github.com/SalvatoreRa/tutorial/blob/main/images/EmotionPrompt2.png?raw=true)
*An example of ReAct prompting. from the [original article](https://arxiv.org/pdf/2210.03629)*

This approach has advantages and disadvantages:
* it reduces hallucinations in comparison with CoT
* is flexible and allows for complex actions
* it depends heavily on the information found and if it does not find relevant results it fails to formulate thoughts
* often works well with large models with good reasoning skills. With small models, it risks meaningless and endless reasoning chains that do not get to the answers


</details>


## Retrieval Augmented Generation (RAG)

<details>
  <summary><b>What is Retrieval Augmented Generation (RAG)?</b></summary>

LLMs can accomplish many tasks but for tasks that require knowledge, they can hallucinate. Especially when the tasks are complex and knowledge-intensive, the model may produce a factually incorrect response. If for tasks that require reasoning we can use the techniques seen above, for tasks that require additional knowledge we can use Retrieval Augmented Generation (RAG).

In short, when we have a query a model looks for the documents that are most relevant in a database. We have an embedder that we use to get a database of vectors, then through similarity search, we search for the vectors that are most similar to the embedding of the query. The documents found are used to augment the generation of our model

![RAG](https://github.com/SalvatoreRa/tutorial/blob/main/images/RAG.png?raw=true)
*from [this article](https://arxiv.org/abs/2307.11760)*


Articles describing in detail:
  * [Cosine Similarity and Embeddings Are Still in Love?](https://levelup.gitconnected.com/cosine-similarity-and-embeddings-are-still-in-love-f9aec98396a4)
  * [Follow the Echo: How to Get a Good Embedding from your LLM](https://levelup.gitconnected.com/follow-the-echo-how-to-get-a-good-embedding-from-your-llm-d243fc2ebcbf)

  
</details>

<details>
  <summary><b>How to select the right chunk strategy for the RAG?</b></summary>

  LLMs have a context length, this is also true for embedded components in the RAG. This means that documents must be divided into chunks. **Chunking** refers to the step where a body of documents is divided into different, more manageable chunks. 

As you can see from HuggingFace's [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), different models have different context lengths, thus a maximum number of tokens they can take as input. This means that a chunk can have a maximum number of tokens.

![RAG](https://github.com/SalvatoreRa/tutorial/blob/main/images/MTEB Leaderboard.png?raw=true)
*from [here](https://huggingface.co/spaces/mteb/leaderboard)*

The chunking strategy also has an impact on model performance: 
* **Relevance and Precision.** Chunks that are too large may contain irrelevant information and information that is too diluted, conversely, if too small relevant information may not be retrieved.
* **Efficiency and Performance.** Larger chunks require more process time and more computational resources. This has an impact on inference. Similarly, select too many small chunks.
* **Generation.** The quality of the answer comes from the information found, so it is important to find the right chunks.
* **Scalability.** Document corpora can be very large, so choosing the right strategy is critical

For this, several strategies have been developed for an efficient chunking system: 
* **Naive Chunking**. The simplest strategy is in which an arbitrary number of tokens are chosen.
* **Semantic Chunking**. In this case, instead of an arbitrary number of tokens, the split is conducted in accordance with semantic rules. Basically, we split chunks when a sentence ends. This can also be achieved with standard libraries such as NLTK and Spacy. A variation on the theme is to search for sentences that are semantically similar (using cosine similarity, for example) and then put them together consecutively.
* **Compound Semantic Chunking**. is an approach derived from the previous one and developed to avoid chunks that are too short. The main difference is that the sentences are concatenated until a certain threshold is reached.
* **Recursive Chunking**. A strategy exploited for structured documents, such as HTML documents, in which the presence of HTML tags is exploited to split. it is recursive because various tags are exploited to conduct various rounds of separation.
* **Specialized chunking**. Similar to the previous one but focused on a particular type of data, such as Markdown, Latex, or code.
* **Context-Enriched**. In this case, the goal is to have chunks where we have useful information and meaningful summaries. There are variants where we generate summaries of different chunks and calculate the similarity between queries and this summary.

![RAG](https://github.com/SalvatoreRa/tutorial/blob/main/images/ChunkViz1.png?raw=true)

![RAG](https://github.com/SalvatoreRa/tutorial/blob/main/images/ChunkViz2.png?raw=true)
*example of chunking of a document, from [here](https://chunkviz.up.railway.app/)*
*from [here](https://huggingface.co/spaces/mteb/leaderboard)*

The choice of chunking depends on several factors. The first is clearly which embedding model you use. If for performance reasons the embedder can only take 512 tokens, the chunking strategy must consider this maximum of tokens. According to the origin of the document, a strategy that respects the document should be chosen. The paragraph division of a document is an important element, so it makes sense to choose a strategy that respects it. Just as it is important to divide the code into chunks that are meaningful. In any case, the best strategy is to test different chunk sizes and strategies and evaluate the results.

</details>

<details>
  <summary><b>How to evaluate a RAG system?</b></summary>

The **evaluation of a RAG application** should take several factors into account. In fact, the main factors affecting the performance of an RAG are: 
* **LLM.** Obviously LLM chosen will influence both how the context is used and the generation.
* **The prompt.** The prompt is more critical than you think and influences how the model deals with the task.
* **The RAG components.** The model chosen as the embedder, whether there are added components, chunking strategy are all elements that influence the outcome.
* **Data quality.** Obviously if the database contains low-quality data this will have a negative impact.

Because we have several factors, we can do a qualitative analysis of the responses, but it would be better to have more quantitative methods and libraries that make the process more automatic and faster. In any case, the perfect RAG should follow two principles: 
* In retrieval it should find all the relevant data but only the relevant data (inclusive but no frills).
* In generation, LLM should be able to synthesize the documents found and resolve if there is a conflict between his knowledge and what he found in the documents


Evaluating RAG application may seem like a difficult task pero there are already Python libraries with this function, for example, RAGAS:

  *"Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. RAG denotes a class of LLM applications that use external data to augment the LLM’s context. There are existing tools and frameworks that help you build these pipelines but evaluating it and quantifying your pipeline performance can be hard. This is where Ragas (RAG Assessment) comes in." --[source](https://github.com/explodinggradients/ragas)*

  The interesting point about RAGAS is that it is reference-free, i.e., one does not need a reference dataset but exploits under the hood an LLM.  RAGAS provides several metrics to assess the quality of an RAG pipeline: 
  * **Context recall.** Evaluates signal-to-noise ratio, basically whether relevant elements are higher in context (the most important chunks should be higher ideally).
  * **Context precision.** Whether all relevant information has been found.
  * **Faithfulness.** measures the factual consistency of the generated answer, and claims in the answer must be able to be supported by the context.
  * **Answer Relevance.** Focuses on understanding how relevant an answer is to the question in the prompt.

There are also other libraries, such as [TruLens](https://www.trulens.org/), which focuses specifically on retrieval relevance. This library calculates the percentage of sentences in the retrieved documents that is relevant to the question.

A limitation of these approaches is that they rely on the assumption that an LLM knows how to evaluate retrieval and knows enough about the question. This assumption is difficult to justify if our RAG deals with technical, complex documents or private data. 

Moreover, LLMs can have a positional bias, which complicates the evaluation when there are different documents:

*"In this paper, we take a sober look at the LLMsas-evaluator paradigm and uncover a significant
positional bias. Specifically, we demonstrate that GPT-4 exhibits a preference for the first displayed candidate response by consistently assigning it higher scores, even when the order of candidates is subtly altered." from [here](https://arxiv.org/pdf/2305.17926.pdf) *

![RAG evaluation](https://github.com/SalvatoreRa/tutorial/blob/main/images/LLM_evaluation_positional_bias.png?raw=true) *from [here](https://arxiv.org/pdf/2305.17926.pdf)*

In these cases, our human assessment should be complemented by human analysis.

Suggested lectures:

* [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
* [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
* [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)


</details>
