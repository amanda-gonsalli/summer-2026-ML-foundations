# Introduction to Machine Learning + Basic Concepts
Machine learning is defined as giving a computer the ability to learn without being explicitly programmed. There are different tools for Machine Learning, the most common being **supervised learning**.

As written by Kevin P. Murphy, in his book "Machine Learning: A Probabilistic Perspective":
> "In particular, we define machine learning as a set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty (such as planning how to collect more data!)."

Overall, the topics covered will be the following:

## 1. Supervised learning
You're given a dataset (X,Y), your goal is to learn a mapping from X to Y. 
We want to learn a function $f: X \to Y$, so that it can be generalized and applied to future inputs as well.
There are different types of functions to be mapped, they can be continuous or not continuous.

* **Regression algorithms:** Y (the output) is a real number

* **Classification algorithms:** Y (the output) is a discrete number (ex: binary output)

Usually, there are multiple variables, that is, you have more than one input. In this case, X is a vector, its dimension being the number of parameters (inputs) given. If there are $n$ features, then $X \in \mathbb{R}^n$

## 2. Machine Learning Strategy (Learning Theory)
Make decisions in a systematic manner to optimze your results.

## 3. Deep Learning
Large models with many layers that learn complex features automatically.

## 4. Unsupervised learning
Uses unlabeled data.
You're given inputs and no outputs. Your goal is to find patterns in the inputs, group similar inputs, and find relationships.
Applications: 
* Given a set of genes (input), map them to individual characteristics
* Cocktail Party Problem: Given microphone recordings from multiple people's voices (overlapping voices), separate the voice from different people talking

## 5. Reinforcement learning
Train the model to find the optimal behavior
When a performance is good, give good reinforcement (reward signal). When the performance is bad, give bad reinforcement
Application: computers playing games by themselves

## Takeaway
Machine learning is fundamentally about learning patterns from data to generalize beyond observed examples.
