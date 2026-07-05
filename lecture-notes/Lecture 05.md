# GDA & Naive Bayes

## Discriminative VS Generative Learning Algorithms
So far, all algorithms we discussed are discriminiative. Suppose we have a binary dataset. Logistic regression, for instance, would use gradient descent to search for a line that separates negative and positive examples.

Generative algorithms look at each class separately. They build a model for each class and define which region of the input space corresponds to each class. Given a new input, the algorithm would compare its features to those of the members of each class and classify the new input as the class whose features most match the input's. Rather than looking for a way to separate classes, generative algorithms create individual models for the classes.

### Formal comparison

Discriminative models learn $P(y|x)$, which depends on $h_\theta(x)$. It learns a mapping from $x$ to $y$.

Generative algorithms learn $P(x|y)$ (given the class, what are the features?). Also learns $P(y)$ which is the class prior (a label given before being given the features).

### Bayes Rule

Faced with a new input, you define its class using the following rule:
$$
P(y = 1|x) = \frac{P(x|y = 1)P(y = 1)}{P(x)}
$$

Where

$$
P(x) = P(x|y = 1)P(y = 1)+P(x|y = 0)P(y = 0)
$$

## Gaussian Discriminant Analysis

Suppode $x\in\mathbb{R}^n$ (drop the $x_0 = 1$ convention) and assume $P(x|y)$ is Gaussian.

Given $z\in \mathbb{R}^n$, the probability density function for a gaussian is:

$$
P(z) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2} (z -\mu)^T\Sigma^{-1}(z-\mu)\right)
$$

Being $\mu$ the mean (n dimensional) and $\Sigma$ the covariance (n x n dimensional)

    Note on covariance:

    The main diagonal controls how spread the probability is (smaller values make it less spread and higher around the mean).
    The off-diagonal controls the correlation bewteen features (positive values indicate positive correlation)

## GDA model
The pdf for all labels are Gaussian

$$
P(x|y =0) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2} (x -\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

$$
P(x|y =1) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2} (x -\mu)^T\Sigma^{-1}(x-\mu)\right)
$$

For binary classification, 

$$
P(y) = \phi^y(1-\phi)^{1-y}
$$

The parameters of a GDA model are $\mu_0, \mu_1 \in \mathbb{R}^n$, $\Sigma \in \mathbb{R}^{n \,x\,n}$ and $\phi\in \mathbb{R}$. This means that, usually, all classes have the same covariance, but different means.

The algorithm learns those parameters, then at test time it uses Bayes rule to find the class of the new input.

We have the training set $\{x^{(i)}, y^{(i)}\}_{i = 1}^m$. We will fit the parameters by maximing the **Joint Likelihood**.

Define the likelihood of the parameters as follows:

$$
\mathcal{L}(\phi, \mu_0, \mu_1, \Sigma) = \prod_{i = 1}^m P(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) = \prod_{i = 1}^m P(x^{(i)}| y^{(i)})P(y^{(i)})
$$

As compared to the discriminative model, which maximes the **conditional likelihood**

$$
\mathcal{L}(\theta) = \prod_{i = 1}^m P(y^{(i)}|x^{(i)};\theta)
$$

Use MLE (Maximum Likelihood Estimation), you get:

$$
\phi =\frac{\sum_{i = 1}^m y^{(i)}}{m} = \frac{\sum_{i=1}^m \mathcal{I}\{y^{(i)}=1\}}{m}\\
-------------------\\

\mu_0 = \frac{\sum_{i = 1}^m\mathcal{I}\{y^{(i)} = 0\}x^{(i)}}{\sum_{i = 1}}\\

-------------------\\
\mu_1 = \frac{\sum_{i = 1}^m\mathcal{I}\{y^{(i)} = 1\}x^{(i)}}{\sum_{i = 1}^m\mathcal{I}\{y^{(i)} =1\}} \\

-------------------\\

\Sigma = \frac{1}{m} \sum_{i = 1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
$$

Note: $\mathcal{I}(\text{true}) = 1$, $\mathcal{I}(\text{false}) = 0$

### Prediction
$$
\arg\max_y P(y|x) = \arg\max_y \frac{P(x|y)P(y)}{P(x)} \\
\arg\max_y P(y|x) = \arg\max_y P(x|y)P(y)
$$

Notation: $\min_z(z-5)^2 = 0$ and $\arg \min_z(z-5)^2 = 5$

Note that $P(x)$ is a constant (doesn't change the $\max$), so it was removed.

## Comparison: GDA and Logistic Regression

For fixed parameters, plot the following as a function of $x$:

$$
P(y=1|x;\phi, \mu_0,\mu_1,\Sigma) =  \frac{P(x|y = 1; \mu_1, \Sigma)P(y = 1;\phi)}{P(x; \phi, \mu_0,\mu_1,\Sigma)}
$$

The first term can be evaluated by the Gaussian density of mean $\mu_1$ and covariance $\Sigma$. For fixed parameters, this is a constant. The second term is a Brenoulli, the probability of y being 1 given the parameter $\phi$ is $\phi$ itself. Similarly, the denominator depends on known numbers. Hence, we can calculate this ratio to find a number for the probability of y being y given x.

The plot turns out to be the sigmoid function, just like in Logistic Regression. However, the two algorithms generate two different decision boundaries.

### When to use generative and when to use discriminative models?
GDA assumes $x$ is Gaussian and $y$ is Bernoulli.

Logistic regression uses $P(y = 1|x) = \frac{1}{1+e^{-\thetaˆTx}}$, and also sets $x_0 = 1$.

The assumptions from DGA imply that $P(y = 1|x)$ follows the sigmoid function for $-\theta^Tx$. The contrary, however, is not true. This means that a GDA uses stronger assumptions than Logistic Regression. Therefore, if the assumptions are generally correct, GDAs do better, if the assumptions are not so correct, then GDA does poorly.

Now, assume that x is Poisson ($\lambda_0, \lambda_1$) and that $y$ is Berboulli. Again, this implies that $P(y = 1|x) = \frac{1}{1+e^{-\theta^Tx}}$.

In reality, this is true for any Generalized Linear Model where $x$ given $y =1$ and $y = 0$ both come from the same exponential family distribution.

In Linear Regression, you don't need to know if the data is Gaussian or Poisson or etc and the model works fine. For GDA, however, you need to know the tyoe of the data beforehand.

You have two main options: stronger assumptions with less data, or weaker assumptions with more data. GDA is also more efficient and faster.

## Naive Bayes

How to get a feature vector $x$?

Consider a spam detector program. First, it needs to take an email and turn it into a vector $x$. One way of doing this is by selecting the most recurring words in the training set and have 1 for words in the email and 0 for those that are not. Therefore, assuming we have $n$ words, we now have  $x\in\{0,1\}^n$.

Want to model $P(x|y), P(y)$.

Assume the $x_i$'s are conditionally independent given $y$, that is:

$$
P(x_1, x_2, \dots, x_n|y ) = P(x_1|y)P(x_2|y) \dots, P(x_n|y) = \prod_{i = 1}^nP(x_i|y)
$$

The parameters of Naive Bayes are the following:

$$
\phi_{j|y = 1} = P(x_j = 1 | y = 1) \\
\phi_{j|y = 0} = P(x_j = 1 | y = 0) \\
\phi_y =P(y = 1)
$$

For $j$ in $[1,n]$

Write out the Joint Likelihood:

$$
\mathcal{L}(\phi_y, \phi_{j | y}) = \prod_{i = 1}^m P(x^{(i)}, y^{(i)};\phi_y, \phi_{j|y})
$$

Then MLE gives:

$$
\phi_y = \frac{\sum_{i = 1}^m \mathcal{I}\{y^{(i)} = 1\}}{m}\\
\phi_{j|y = 1} = \frac{\sum_{i = 1}^m \mathcal{I}\{x^{(i)}_j = 1, y^{(i)} = 1\}}{\sum_{i = 1}^m \mathcal{I}\{y^{(i)} = 1\}}
$$
