

# Preliminary: a refresh

To begin this journey, if we do not feel confident about the pillars of machine learning, I recommend reviewing some of the fundamental concepts that will return often as we tackle Large Language Models.

To best understand how a Large Language Model (LLM) works, it is important to have a good understanding of the mathematical and machine learning principles behind it. it is recommended to focus on three axes: 

* **Mathematical preliminaries** 
* **Python basis**
* **NLP basis**

## Mathematical basis
There are several aspects of mathematics that can be useful in understanding and planning an LLM. Generally, these prerequisites are common to both machine learning and deep learning. 
* **Linear algebra** is the basis for understanding most deep learning concepts. Linear algebra is used for data manipulation, transformation and modeling. knowledge of vectors, matrices and linear equations is crucial.
* **Calculus** is used to describe the behavior of algorithms. Concepts such as gradient descent and backpropagation are critical to understanding how a neural network (and thus an LLM) is trained.
* **Statistics** is used both to understand the dataset (initial analysis, visualization, obtaining insights) but also to study the behavior of the model. In general, statistics is the basis for understanding the relationship between dependent and independent variables, allowing us to investigate what the model learns
* **Probability theory** allows us to efficiently represent the degree of uncertainty in knowledge. This will be intriguingly related to how LLMs predict the next token when they generate text.



### Resources 

**High level Resources**

* [3Blue1Brown - The Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): This YouTube channel provides visually intuitive explanations of mathematical concepts, including linear algebra and neural networks.
* [StatQuest with Josh Starmer - Statistics Fundamentals](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9): This YouTube series offers clear and engaging explanations of fundamental statistical concepts. Hosted by Josh Starmer, the videos use simple visuals and straightforward language to make topics like probability, regression, hypothesis testing, and statistical distributions accessible to learners at all levels. 
* [AP Statistics Intuition by Ms Aerin](https://automata88.medium.com/list/cacc224d5e7d):  This Medium series by Ms. Aerin provides insightful explanations of AP Statistics concepts. The articles focus on building a deep understanding of statistical principles through intuitive explanations, making complex topics more approachable..
* [Immersive Linear Algebra](https://immersivemath.com/ila/learnmore.html): This interactive online textbook offers a comprehensive and engaging approach to learning linear algebra. It combines clear explanations with dynamic visualizations and interactive exercises to help readers develop a deep understanding of key concepts such as vectors, matrices, determinants, and eigenvalues..
* [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra): This comprehensive course on Khan Academy covers the fundamental concepts of linear algebra. It includes topics such as vector spaces, matrices, determinants, eigenvalues, and linear transformations. 
* [Khan Academy - Calculus](https://www.khanacademy.org/math/calculus-1): This extensive course on Khan Academy covers key topics in calculus, including limits, derivatives, integrals, and the fundamental theorem of calculus. Through a series of video lessons, interactive exercises, and step-by-step explanations, learners can develop a strong understanding of calculus concepts.
* [Khan Academy - Probability and Statistics](https://www.khanacademy.org/math/statistics-probability): This detailed course on Khan Academy covers essential concepts in probability and statistics. Topics include descriptive statistics, probability distributions, random variables, hypothesis testing, and inferential statistics. 

**In-depth level Resources**

* [Goodefellow I. - Deep learning book cap II](https://www.deeplearningbook.org/contents/linear_algebra.html): provides an essential introduction to linear algebra. It covers key concepts such as vectors, matrices, matrix multiplication, linear transformations, and eigenvalues. 
* [Goodefellow I. - Deep learning book cap III](https://www.deeplearningbook.org/contents/prob.html) delves into the fundamentals of probability and information theory. It covers topics such as probability distributions, marginal and conditional probabilities, independence, Bayes' theorem, entropy, and mutual information.: 
* [Goodefellow I. - Deep learning book cap IV](https://www.deeplearningbook.org/contents/numerical.html): focuses on numerical computation techniques that are vital for implementing and optimizing deep learning algorithms. It covers essential topics such as floating-point arithmetic, optimization algorithms, numerical stability, and the use of software tools for efficient computation.
* [Jentzen A. - Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory](https://arxiv.org/abs/2310.20360): provides a comprehensive mathematical foundation for understanding deep learning algorithms. A book for advancer users who want to delve in the mathamatical understanding of deep learning concepts (600 pages)

## Python basics

Python is the language of machine learning; most of the algorithms and libraries speak in Python. Although there are resources in other languages, Python is particularly dominant when it comes to neural networks and consequently LLM. A fair amount of knowledge of Python is recommended in order to be able to successfully utilitize the algorithms.

* **Python basis**. it is recommended to have an understanding of Python basics, syntax and the fundamental concepts behind using Python (object oriented programming, functions and so on). 
* **Data science with Python** some libraries are omni present, especially if you want to manipulate data or plot results it is best to have a knowledge of libraries such as Matplotilib, Numpy or Pandas. 
* **Machine learning concepts** libraries such as scikit-learn are the basis of how to approach machine learning. Many of the algorithms there are then used for more complex concepts later. In general, knowledge of classical machine learning algorithms is important. Those who want to approach the study of LLMs will find it beneficial to know concepts such as: dimensionality reduction, classification, regression.

**Resources**

* [Real Python](https://realpython.com/): Real Python is a comprehensive online resource dedicated to teaching Python programming. It offers a wide range of tutorials, articles, and video courses covering everything from basic Python syntax to advanced topics like web development, data science, and automation. 
* [freeCodeCamp - Learn Python](https://www.youtube.com/watch?v=rfscVS0vtbw): Long video that provides a full introduction into all of the core concepts in Python.
* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/): this online resource serves as a thorough guide to using Python for data science. It encompasses key topics such as data manipulation with Pandas, numerical computing with NumPy, data visualization with Matplotlib, and machine learning with Scikit-Learn.  
* [freeCodeCamp - Machine Learning for Everybody](https://youtu.be/i_LwzRVP7bg): his YouTube video by freeCodeCamp provides an in-depth introduction to machine learning accessible to a broad audience. The tutorial covers fundamental concepts such as supervised and unsupervised learning, algorithms like linear regression, decision trees, and neural networks, and practical applications of machine learning. 
* [Udacity - Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120): This online course by Udacity offers a foundational introduction to machine learning. It covers essential topics such as supervised and unsupervised learning, decision trees, clustering, regression, and more.

**In-depth resources**

* [Goodefellow I. - Deep learning book cap V](https://www.deeplearningbook.org/contents/ml.html): This chapter discusses fundamental concepts such as supervised and unsupervised learning, the importance of training data, and the role of models in predicting outputs from inputs. It explores various algorithms, the balance between bias and variance, and techniques for evaluating and improving model performance. 

## Neural net basics

Large language models are basically neural networks with many more parameters. To best understand them, it is important to know the basics of deep learning and how neural networks work.

![neuron](https://github.com/SalvatoreRa/tutorial/blob/main/images/neuron.png?raw=true)
*from Wikipedia*

Among the fundamentals for understanding neural networks, the concept of **artificial neuron** is important. How these can then be combined into a layer and how these can then be added on top of each other. At a mechanistic look, one can see how these are formed by matrices of weights and biases. In addition, it is important to understand the ruole of activation functions (without a nonlinear activation function one could not have deep learning). 

Once we have built our first neural network, we need to understand how to train it. Neural networks are trained by backpropagation, which allows error propagation to conduct the weights update. During training we calculate the error using an error function (different losses are for different tasks, cross-entropy for classification, mean squared error for regression and so on). In addition, optimization of weights is conducted by algorithms such as gradient descent and its variations (SGD, Adam, RMSprop and so on).

![gradient descent](https://github.com/SalvatoreRa/tutorial/blob/main/images/GradientDescentGradientStep.svg?raw=true)
*from [here](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)*

Special attention should be paid to the concept of overfitting. Overfitting is indeed one of the most important challenges to neural network training. In fact, a model trained on the training set can then perform poorly on new data (lack of generalization). Therefore, a whole range of regularization techniques have evolved to reduce overfitting (L1/L2 regularization, dropout, data augmentation. One intriguing phenomenon is grokking when an apparently overfitting network begins to generalize (delayed generalization)

![overfitting](https://github.com/SalvatoreRa/tutorial/blob/main/images/overfitting_nn_curve.png?raw=true)
*from [here](https://arxiv.org/abs/1812.11118)*

**High level resources**

* [3Blue1Brown - But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk):  This video by 3Blue1Brown provides a visually intuitive explanation of neural networks. Using engaging animations and clear explanations, it introduces the basic concepts of how neural networks function, including the structure of neurons, layers, and how data is processed through the network.
* [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c): This YouTube video by freeCodeCamp offers an in-depth introduction to deep learning. Covering fundamental concepts such as neural networks, activation functions, backpropagation, and more, the tutorial provides practical examples and coding exercises to help learners understand and apply deep learning techniques. 
* [Fast.ai - Practical Deep Learning](https://course.fast.ai/): This online course by Fast.ai provides a hands-on approach to learning deep learning. It covers practical techniques and applications, focusing on real-world projects and examples. The course includes topics such as image classification, natural language processing, and collaborative filtering, and emphasizes understanding the underlying principles and code implementations. 
* [Patrick Loeber - PyTorch Tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4):  This YouTube playlist by Patrick Loeber offers a series of tutorials focused on PyTorch, a popular deep learning framework. The tutorials cover a wide range of topics, including the basics of PyTorch, building neural networks, training models, and implementing various deep learning algorithms.
* [Crash Course in Deep Learning](https://gpuopen.com/learn/deep_learning_crash_course/) - The creation and application of multi-layer perceptrons (MLPs), a kind of fully connected neural network used in deep learning, are covered in this article.

**In depth resources**

* [Goodefellow I. - Deep learning book cap VI](https://www.deeplearningbook.org/contents/mlp.html): The chapter discusses deep feedforward networks. It provides are in-depth explanation of the fundamental deep learning models used for approximating functions by mapping inputs to outputs through layers of computations. Key concepts include the structure of networks, training using gradient-based methods, and the importance of hidden layers and activation functions.
* [Goodefellow I. - Deep learning book cap VII](https://www.deeplearningbook.org/contents/regularization.html): the chapter overs regularization strategies in deep learning, aiming to reduce test error and prevent overfitting. It details various methods, such as parameter norm penalties (L2 and L1 regularization), ensemble methods, and techniques to balance bias and variance. The focus is on how these strategies can be applied to neural networks, discussing the trade-offs between complexity and generalization, and highlighting specific methods like weight decay and sparsity-inducing penalties.
* [Goodefellow I. - Deep learning book cap VIII](https://www.deeplearningbook.org/contents/optimization.html): This chapter discusses optimization techniques for training deep learning models. It covers the differences between machine learning optimization and pure optimization, highlighting challenges like ill-conditioning and the need for specialized algorithms. It also explains empirical risk minimization, surrogate loss functions, early stopping, and batch versus minibatch algorithms.
* [Neural Networks and Neural Language Models](https://web.stanford.edu/~jurafsky/slp3/7.pdf) - Dan Jurafsky and James H. Martin. This chapter of Speech and Language Processing describes in details feed-forward neural networks, specifically applied to NLP. It is great resources and easy to understand.

## Natural Language Processing

![NLP evolution](https://github.com/SalvatoreRa/tutorial/blob/main/images/evolution_nlp.png?raw=true)
*From [here](https://arxiv.org/pdf/2303.18223)*

Natural Language Processing (NLP) is that branch of artificial intelligence that deals with understanding human language. Humans express themselves through language and convey complex information. To be able to understand the various nuances of human language, sophisticated algorithms have been developed. 

Text in its natural form is difficult for a machine to digest. Therefore, it is important to understand how to process a text (stemming, lemmatization, cleaning) and reduce it into units (tokenization). Once this is done one must then transform it into a vector representation that can then be used by computers (TF-IDF, bag-of-words, n-grams, embeddings). Special attention should be given to word embeddings because they learn dense vectors that represent the similarity in meaning of various words.

![embedding visualization](https://github.com/SalvatoreRa/tutorial/blob/main/images/embedding_visualization.png?raw=true)
*visualization of an embedding. From [here](https://ai.google.dev/gemini-api/tutorials/clustering_with_embeddings)*

The recurrent nature of text sequences requires special adaptation of neural networks. For example, Recurrent Neural Networks (RNNs) are neural networks that have an additional vector to hold memory for previous elements of the sequence (and derived variants such as LSTM and GRU). the lack of parallelization of RNNs, vanishing gradient, and modeling long dependencies are some of the challenges that later led to the transformer. 

![RNN structure](https://github.com/SalvatoreRa/tutorial/blob/main/images/Recurrent_neural_network_unfold.png?raw=true)
*from Wikipedia*

**High level Resources**

* [RealPython - NLP with spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/): The tutorials cover a wide range of topics, including the basics of PyTorch, building neural networks, training models, and implementing various deep learning algorithms. 
* [Kaggle - NLP Guide](https://www.kaggle.com/learn-guide/natural-language-processing): This comprehensive guide on Kaggle provides a structured learning path for natural language processing (NLP). It includes tutorials, hands-on exercises, and practical examples to cover essential NLP concepts and techniques. Topics include text preprocessing, sentiment analysis, text classification, and more.
* [Jay Alammar - The Illustration Word2Vec](https://jalammar.github.io/illustrated-word2vec/): This blog post by Jay Alammar offers a visually engaging explanation of the Word2Vec algorithm, a popular method for creating word embeddings in natural language processing.
[Stackoverflow - An intuitive introduction to text embeddings](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/) - t covers how embeddings help in understanding semantic relationships between words and improve tasks like search engines, recommendation systems, and natural language processing.
* [Visualize word embeddings](https://projector.tensorflow.org/) - The TensorFlow Embedding Projector is a tool for visualizing high-dimensional data. It allows users to load and explore embeddings using techniques like PCA, t-SNE, and UMAP.
* [Jake Tae - PyTorch RNN from Scratch](https://jaketae.github.io/study/pytorch-rnn/): This tutorial by Jake Tae provides a comprehensive guide to building Recurrent Neural Networks (RNNs) from scratch using PyTorch. It covers the fundamental concepts of RNNs, including how they process sequential data, and provides step-by-step instructions for implementing an RNN model in PyTorch.
* [colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/):  This insightful blog post by Christopher Olah provides a thorough and accessible explanation of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) particularly effective for sequence data. The post breaks down the complex inner workings of LSTMs with detailed diagrams and intuitive explanations, covering key components like cell states, gates, and the flow of information.

**In-depth resources**

* [Goodefellow I. - Deep learning book cap X](https://www.deeplearningbook.org/contents/rnn.html): This chapter explains recurrent neural networks (RNNs) and recursive networks for processing sequential data. It describes the structure and operation of RNNs, emphasizing parameter sharing and unfolding computational graphs. The content includes different RNN architectures, training methods, and applications in tasks like language modeling and sequence prediction.
* [A Guide on Word Embeddings in NLP](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp) - This article covers the fundamentals of word embeddings, their importance, and applications in natural language processing. It explains how word embeddings convert words into vectors, enabling machines to process text data more effectively.
* [RNNs and LSTMs](https://web.stanford.edu/~jurafsky/slp3/9.pdf) - Dan Jurafsky and James H. Martin. This chapter of Speech and Language Processing describes in details RNN and LSTMs. It describes also the training of these models. It is great resources and easy to understand.

# LLM basic

## The LLM architecture

An LLM is basically a transformer with many more parameters. In order to understand how an LLM works, it is important to understand at least at a high level how a transformer works. Especially since the attention mechanism was so revolutionary.

The transformer consists of a few basic components:
* A **tokenizer** that allows text to be mapped to integers. The tokenizer is an important component that is often the cause of some unexpected behavior of the 
* A **embedding layer** that allows learning a vector and contextual representation of text.
* a **transformer-block**, this is the beating heart of the transformer. The model is composed of several layers stacked on top of each other. Each transformer block is composed of **multi-head self-attention** followed by **batch normalization** **feed-forward layer**. Also present in the block are residual connection
* The last layer to decode and generate text.
* the transfomer consists of an encoder and a decoder, but there are models today that are encoders only or decoders only. 

![the transformer architecture](https://github.com/SalvatoreRa/tutorial/blob/main/images/transformer_structure.png?raw=true)
*from [original article](https://arxiv.org/pdf/1706.03762v5)*

A particular point is to include multi-head self-attention (MHSA), because this is important for the transformer. MHSA allows the transformer to be so powerful, in fact modeling the relationships between the various tokens. Also, since there are multiple attention heads at each layer we can learn for each tokens different representations. Mechanistic studies show that the transformer learns a hierarchical representation, and the various layers go on to learn increasingly complex representations of the text.

![multi-head self attention scheme](https://github.com/SalvatoreRa/tutorial/blob/main/images/multi_head_self_attention.png?raw=true)
*from [original article](https://arxiv.org/pdf/1706.03762v5)*

**High level resources**

Here a list of resources that can give a great grasp of inner mechanism of a transformer. I add also different resources about self-attention.

* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar. The most famous and illustrated explanation of the transformer.
* [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Jay Alammar. GPT-2 is the grandfather of current LLMs (autoregressive model), this detailed blog post explains in details the model
* [Transformer explainer](https://poloclub.github.io/transformer-explainer/) - another great and interactive visualization of the transformer (based on GPT2). It allows to understand the inner working of a transformer. Check also the associated [paper](https://arxiv.org/abs/2408.04619)
* [Visual intro to Transformers](https://www.youtube.com/watch?v=wjZofJX0v4M&t=187s) - 3Blue1Brown. Another great resource about an introduction on transformer in a video
* [LLM Visualization](https://bbycroft.net/llm)  Brendan Bycroft. 3D visualization to dig inside the transformer
* [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy. Karpathy explain in this video how to implement GPT model from scratch (very useful at the programming side).
* [Coding the Self-Attention Mechanism of LLMS From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) - Sebastian Raschka. Great blog post where is explained how to code self-attention (and cross-attention) in PyTorch
* [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng. A formal description of attention
* [Decoding Strategies in LLMs](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html). An introduction to the different strategies for decoding. 

**Theroretical resources**

A list of resources about a more formal introduction on transformers and attention. I strongly suggest these resources to whom want to dig and better understand in details.

* [Transformers and Large Language Models](https://web.stanford.edu/~jurafsky/slp3/10.pdf) - Dan Jurafsky and James H. Martin. This chapter of Speech and Language Processing describes in details the self-attention, the training, and the transformer. It is great resources and easy to understand.
* [Fine-tuning and Masked Language Models](https://web.stanford.edu/~jurafsky/slp3/11.pdf) - Dan Jurafsky and James H. Martin. This chapter is focused on the BERT like transformer and how to fine-tune a model. Both chapters are easy and very insightful
* [Attention is all you need](https://arxiv.org/abs/1706.03762) - The seminal paper that introduced the transformer

## Pre-training of the LLM

* **Data curation**
* **Scaling laws**
* **High-Performance Computing**

**High level resources**

* [LLMDataHub](https://github.com/Zjh-819/LLMDataHub)  - It curates and organizes various datasets essential for training LLMs, particularly those used in chatbot development and instruction-based fine-tuning. The repository includes details such as dataset types, languages, sizes, and descriptions
* [Training a causal language model from scratch](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt)- a guide by HuggingFace in how to pre-train a GPT-2 like model from scratch
* [TinyLlama](https://github.com/jzhang38/TinyLlama) - how to train TinyLlama, a smaller and more efficient version of the LLaMA (Large Language Model). This will help to learn how LLaMA has been trained.
* [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) - BLOOM3 is a GPT3 like model (but an institutional effort to create an open-source model). The team behind the development provided details on the model’s capabilities, architecture, training data, and usage guidelines. 
* [LLM 360](https://www.llm360.ai/) - it offer tools, resources, and information related to large language models (LLMs), likely focusing on their development, application, and best practices.
* [nanoGPT](https://github.com/karpathy/nanoGPT) - by Karpathy provides a minimalistic implementation of GPT (Generative Pretrained Transformer) for educational purposes and small-scale experiments. [the video where he explains the project](https://www.youtube.com/watch?v=kCc8FmEb1nY)

**Theroretical resources**

* [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) - The article on LessWrong discusses the significant implications of DeepMind's Chinchilla paper, which suggests that large language models are often undertrained relative to their size. The key takeaway is that training data should be scaled up alongside model size to optimize performance efficiently.
* [Understanding LLMs: A Comprehensive Overview from Training to Inference](https://arxiv.org/abs/2401.02038v2) - This paper reviews the evolution of large language model training techniques and inference deployment technologies aligned with this emerging trend. 
* [Challenges and Responses in the Practice of Large Language Models](https://arxiv.org/abs/2408.09416) - A curated list of important questions (and answers) about infrastructure, architecture, data, application.

## Supervised Fine-tuning


**High level resources**
*  [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) - A guide from HuggingFace to how fine-tune a DistilGPT-2 model. HuggingFace provides easy and quick way to fine-tune a model.
* [A Beginner's Guide to LLM Fine-Tuning](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html) - It covers essential concepts, the setup process, and practical steps for fine-tuning models using accessible tools like Google Colab. 
* [The Novice's LLM Training Guide](https://rentry.org/llm-training) - A beginners'guide to key concepts, practical steps, and tools needed for training or fine-tuning LLMs, including setting up datasets, choosing model architectures, and optimizing performance
* [LoRA insights](https://lightning.ai/pages/community/lora-insights/) - An article by Sebastian Raschka about he basics of LoRA, how it can be used to efficiently fine-tune large language models (LLMs), It also highlights community-driven efforts and real-world examples of LoRA being applied in various projects.
* [Fine-Tune Your Own Llama 2 Model](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html) - a step-by-step guide on how to fine-tune a LLaMA 2 model using a Google Colab notebook. The tutorial walks you through setting up the environment, loading the model, preparing the dataset, and executing the fine-tuning process.
* [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff) - a detailed explanation of padding in large language models (LLMs), focusing on its implementation using LLaMA 2.

**Theroretical resources**

* [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/abs/2308.10792) -
* []() -
* []() -

## Alignment & Instruction tuning

**High level resources**

**Theroretical resources**

## The LLM datasets

* [Preparing a Dataset for Instruction tuning](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2) -  Weights & Biases provides a detailed guide on preparing datasets for fine-tuning large language models (LLMs). It covers the importance of high-quality datasets, outlines steps for data collection and cleaning, and emphasizes the need for diverse and representative examples.
* [Generating a Clinical Instruction Dataset](https://medium.com/mlearning-ai/generating-a-clinical-instruction-dataset-in-portuguese-with-langchain-and-gpt-4-6ee9abfa41ae) - Medium article that details a process of creating a dataset for clinical instructions using LangChain and GPT-4. It covers steps including data collection, preprocessing, and leveraging GPT-4 for generating relevant instructions in Portuguese. The guide emphasizes practical implementation and offers insights into handling language-specific challenges in dataset creation.
* [GPT 3.5 for news classification](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f) - Kshitiz Sahay discusses the process of using GPT-3.5 to generate a dataset for fine-tuning the Llama 2 model for news classification tasks. It covers dataset creation, preprocessing steps, and the fine-tuning process, providing practical insights and lessons learned during the project.
* [Dataset creation for fine-tuning LLM](https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing) -  A Google Colab notebook that contains technique to filter a dataset

## Prompt engineering

**High level resources**

**Theroretical resources**

## Evaluation

**High level resources**

**Theroretical resources**

## Optimization

**High level resources**

**Theroretical resources**

## Dimensionality reduction

**High level resources**

**Theroretical resources**

## Security

**High level resources**

**Theroretical resources**

* [Controllable Text Generation for LLMs](https://arxiv.org/abs/2408.12599) - a survey about controlling text generation in LLMs with a focus on consistency, style, helpfulness and safety

## Alternative to the transformer

**High level resources**

**Theroretical resources**

## Multimodal

**High level resources**

**Theroretical resources**

## Emerging trend

**High level resources**

**Theroretical resources**

## Deployment

**High level resources**

**Theroretical resources**

## Retrieval-Augmented Generation (RAG)

**High level resources**
* [Scaling RAG for Big Data](https://ragaboutit.com/scaling-rag-for-big-data-techniques-and-strategies-for-handling-large-datasets/) - a deep dive about techniques and strategies for handling large datasets with RAG systems

**Theroretical resources**

## LLM and Knowledge Graphs

**High level resources**

**Theroretical resources**

* [Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2408.08921) - A survey about GraphRAG. It discusses in depth the workflow (from indexing to generation), applications, evaluation and industry case scenarios.

## Agents

**High level resources**

**Theroretical resources**

