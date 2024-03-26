# Frequently Asked Questions on machine learning and artificial intelligence
## Python tutorial 

![artificial intelligence](https://github.com/SalvatoreRa/tutorial/blob/main/images/nn_brain.jpeg?raw=true)

Photo by [Alina Grubnyak](https://unsplash.com/@alinnnaaaa) on [Unsplash](https://unsplash.com/)

&nbsp;

# Index
* [FAQ on machine learning](#FAQ-on-machine-learning)
* [FAQ on artificial intelligence](#FAQ-on-artificial-intelligence)

&nbsp;

# FAQ on machine learning

&nbsp;

# FAQ on artificial intelligence

<details>
  <summary><b>What is machine learning?</b></summary>
  !
</details>

## Large Language Models



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
  <summary><b>What is Retrieval Augmented Generation (RAG)?</b></summary>

LLMs can accomplish many tasks but for tasks that require knowledge, they can hallucinate. Especially when the tasks are complex and knowledge-intensive, the model may produce a factually incorrect response. If for tasks that require reasoning we can use the techniques seen above, for tasks that require additional knowledge we can use Retrieval Augmented Generation (RAG).

In short, when we have a query a model looks for the documents that are most relevant in a database. We have an embedder that we use to get a database of vectors, then through similarity search, we search for the vectors that are most similar to the embedding of the query. The documents found are used to augment the generation of our model

![RAG](https://github.com/SalvatoreRa/tutorial/blob/main/images/RAG.png?raw=true)
*from [this article](https://arxiv.org/abs/2307.11760)*



Articles describing in detail:
  * [Cosine Similarity and Embeddings Are Still in Love?](https://levelup.gitconnected.com/cosine-similarity-and-embeddings-are-still-in-love-f9aec98396a4)
  * [Follow the Echo: How to Get a Good Embedding from your LLM](https://levelup.gitconnected.com/follow-the-echo-how-to-get-a-good-embedding-from-your-llm-d243fc2ebcbf)

  
</details>

