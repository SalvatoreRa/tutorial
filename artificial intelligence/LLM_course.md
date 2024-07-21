

# Preliminary: a refresh

To begin this journey, if we do not feel confident about the pillars of machine learning, I recommend reviewing some of the fundamental concepts that will return often as we tackle Large Language Models.

To best understand how a Large Language Model (LLM) works, it is important to have a good understanding of the mathematical and machine learning principles behind it. it is recommended to focus on three axes: 

* **Mathematical preliminaries** 
* **Python basis**
* **NLP basis**

## Mathematical basis



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

## Python basics

**Resources**

* [Real Python](https://realpython.com/): Real Python is a comprehensive online resource dedicated to teaching Python programming. It offers a wide range of tutorials, articles, and video courses covering everything from basic Python syntax to advanced topics like web development, data science, and automation. 
* [freeCodeCamp - Learn Python](https://www.youtube.com/watch?v=rfscVS0vtbw): Long video that provides a full introduction into all of the core concepts in Python.
* [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/): this online resource serves as a thorough guide to using Python for data science. It encompasses key topics such as data manipulation with Pandas, numerical computing with NumPy, data visualization with Matplotlib, and machine learning with Scikit-Learn.  
* [freeCodeCamp - Machine Learning for Everybody](https://youtu.be/i_LwzRVP7bg): his YouTube video by freeCodeCamp provides an in-depth introduction to machine learning accessible to a broad audience. The tutorial covers fundamental concepts such as supervised and unsupervised learning, algorithms like linear regression, decision trees, and neural networks, and practical applications of machine learning. 
* [Udacity - Intro to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120): This online course by Udacity offers a foundational introduction to machine learning. It covers essential topics such as supervised and unsupervised learning, decision trees, clustering, regression, and more.

## Neural net basics

**Resources**

* [3Blue1Brown - But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk):  This video by 3Blue1Brown provides a visually intuitive explanation of neural networks. Using engaging animations and clear explanations, it introduces the basic concepts of how neural networks function, including the structure of neurons, layers, and how data is processed through the network.
* [freeCodeCamp - Deep Learning Crash Course](https://www.youtube.com/watch?v=VyWAvY2CF9c): This YouTube video by freeCodeCamp offers an in-depth introduction to deep learning. Covering fundamental concepts such as neural networks, activation functions, backpropagation, and more, the tutorial provides practical examples and coding exercises to help learners understand and apply deep learning techniques. 
* [Fast.ai - Practical Deep Learning](https://course.fast.ai/): This online course by Fast.ai provides a hands-on approach to learning deep learning. It covers practical techniques and applications, focusing on real-world projects and examples. The course includes topics such as image classification, natural language processing, and collaborative filtering, and emphasizes understanding the underlying principles and code implementations. 
* [Patrick Loeber - PyTorch Tutorials](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4):  This YouTube playlist by Patrick Loeber offers a series of tutorials focused on PyTorch, a popular deep learning framework. The tutorials cover a wide range of topics, including the basics of PyTorch, building neural networks, training models, and implementing various deep learning algorithms.

## Natural Language Processing

**Resources**

* [RealPython - NLP with spaCy in Python](https://realpython.com/natural-language-processing-spacy-python/): The tutorials cover a wide range of topics, including the basics of PyTorch, building neural networks, training models, and implementing various deep learning algorithms. 
* [Kaggle - NLP Guide](https://www.kaggle.com/learn-guide/natural-language-processing): This comprehensive guide on Kaggle provides a structured learning path for natural language processing (NLP). It includes tutorials, hands-on exercises, and practical examples to cover essential NLP concepts and techniques. Topics include text preprocessing, sentiment analysis, text classification, and more.
* [Jay Alammar - The Illustration Word2Vec](https://jalammar.github.io/illustrated-word2vec/): This blog post by Jay Alammar offers a visually engaging explanation of the Word2Vec algorithm, a popular method for creating word embeddings in natural language processing. 
* [Jake Tae - PyTorch RNN from Scratch](https://jaketae.github.io/study/pytorch-rnn/): This tutorial by Jake Tae provides a comprehensive guide to building Recurrent Neural Networks (RNNs) from scratch using PyTorch. It covers the fundamental concepts of RNNs, including how they process sequential data, and provides step-by-step instructions for implementing an RNN model in PyTorch.
* [colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/):  This insightful blog post by Christopher Olah provides a thorough and accessible explanation of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) particularly effective for sequence data. The post breaks down the complex inner workings of LSTMs with detailed diagrams and intuitive explanations, covering key components like cell states, gates, and the flow of information.

# LLM basic

**High level resources**

Here a list of resources that can give a great grasp of inner mechanism of a transformer. I add also different resources about self-attention.

* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar. The most famous and illustrated explanation of the transformer.
* [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Jay Alammar. GPT-2 is the grandfather of current LLMs (autoregressive model), this detailed blog post explain in details the model
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