# Locally Weighted & Logistic Regression 

Suppose you have a dataset that is not well fit by a straight line. A quadratic or a square root would be more appropiate, for instance. To do that with linear regresion, one method is defining one of the features of $x$ to be $x^2$, or, similarly, to be $\sqrt{x}$, for these two examples.

## Locally Weighted Regression

Some Machine Learning terms:
* **Parametric learning algorithms**: Fit a fixed set of parameters ($\theta_i$) to data
* **Non-Parametric learning algorithms**: The amount of data/ parameters you need to keep grows (linearly)  with the size of the training set

### Intuition

Localy weighted regression is a non-parametric algorithm. To evaluate $h$ at certain $x$, you look at the training data around the point you wish to make a prediciton on. Fit a line through this region, and this line is used for the prediction.

### Mathemtical definition

Fit $\theta$ to minimize the following adjusted cross-frunction:

$$
J(\theta) = \sum_{i = 1}^M w^{(i)} (y^{(i)} - \theta^Tx^{(i)})^2
$$

Where $w^{(i)}$ is a weight function, usually defined as:

$$
w^{(i)} = \exp \left(-\frac{(x^{(i)} - x)^2}{2 \tau^2}\right)
$$

If $|x^{(i)} - x|$ is small, then $w^{(i)} \approx 1$.

If $|x^{(i)} - x|$ is large, then $w^{(i)} \approx 0$.

This means that the weight for points closer to the point $x$ you're evaluating is larger than the weight for those points further away.

### How to choose $\tau$?

$\tau$ is a hyperparameter called bandwidth, which describes the width of the bell curve. Great $\tau$ values (broader functions) may oversmooth the data.

Locally weighted linear regression is often used when you have many example data with few features.

## Probabilistic interpretation of linear regression

### Why least squares?

Assume $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, where $\epsilon$ is ann error term for unmodelled effects and noise.

Let the error be a normal distribution with mean zero and variance $\sigma^2$. That is, $\epsilon \sim \mathcal{N}(0, \sigma^2)$

This means that the probability density function for the error is the Gaussian distribution:

$$
P(\epsilon^{(i)}) =\frac{1}{\sqrt{2\pi} \sigma} \exp \left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)
$$

This implies that the probability for the result $y^{(i)}$ given the input $x^{(i)}$ and parameterized by $\theta$ is

$$
P(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi} \sigma} \exp \left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)
$$

Or, similarly,

$$
y^{(i)}|x^{(i)};\theta \sim \mathcal{N}(\theta^Tx^{(i)}, \sigma^2)
$$

Note: the semicolon should be read as "parameterized by".

Define the likelihood of the parameter $\theta$ as the probability of all $y$'s given all $x$'s parameterized by $\theta$:

$$
\mathcal{L}(\theta) = P(\vec{y}|x; \theta) = \prod_ {i = 1}^M P(y^{(i)} | x^{(i)};\theta) = \prod_{i = 1}^M \frac{1}{\sqrt{2\pi} \sigma} \exp \left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)
$$

    What's the difference between the likelihood of the parameters and the probability of the data?

    If you hold the data fixed (with training examples), then the probability of the results becomes a function of the parameter. Then, we call it the likelihood of the parameters.

    However, if you fix the parameter and vary the data, then we call it the probability of the data.


### Maximum likelohood estimation (MLE)

Choose $\theta$ to maximize the likelihood $\mathcal{L}(\theta)$. It is easier, however, to mazimize $\log \mathcal{L}(\theta)$. 

Define $l(\theta) = \log \mathcal{L}(\theta)$. Extending this, we have

$$
l = \log \prod_{i = 1}^M \frac{1}{\sqrt{2\pi} \sigma} \exp \left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right) = \sum_{i = 1}^M \left[\log \frac{1}{\sqrt{2\pi} \sigma} + \log \exp \left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)\right]\\
l = M \log \frac{1}{\sqrt{2\pi} \sigma} + \sum_{i = 1}^M -\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
$$

To maximize it, note that the first term is a constant, so we only concern ourselves with the summation term. On the second term, $\sigma$ is also a constant.

Choose $\theta$ to minimize $ \frac{1}{2}\sum_{i = 1}^M(y^{(i)}-\theta^Tx^{(i)})^2$, since it has a negatvie sign in the expression above. Note that we have arrived back to the cross-function. **Therefore, to maximize the log likelihood is equivalent to minimizing $J(\theta)$**.

## Classification Problems: Logistic Regression

How to solve a Binary Classification with $y\in\{0, 1\}$ using logistics regression?

We want $h_\theta(x) \in [0,1]$. Define the "sigmoid"/ "logistic" function as follows:

$$
g(z) = \frac{1}{1 + e^{-z.}}
$$

Note that this function is always between 0 and 1

Then, we write the hypothesis function:

$$
h_\theta (x) = g(\theta^Tx) = \frac{1}{1 + e^{-\theta^Tx}}
$$

$y$ can be only 0 or 1, so if $P(y = 1 | x; \theta) = h_\theta(x)$, then $P(y = 0|x; \theta) = 1 - h_\theta(x)$. This can be written as:

$$
P(y|x; \theta) = h_\theta(x)^y (1-h_\theta(x))^{1-y}
$$

Consider now $\mathcal{L}(\theta) = P(\vec{y}|x; \theta)$. Expanding into now known expressions:

$$
\mathcal{L}(\theta) =  \prod_ {i = 1}^M P(y^{(i)} | x^{(i)};\theta) = \prod_ {i = 1}^M h_\theta(x^{(i)})^{y^{(i)}} (1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$

Again, compute the log likelihood $l(\theta) = \log \mathcal{L}(\theta)$

$$
\log \mathcal{L}(\theta) = \log \prod_ {i = 1}^M h_\theta(x^{(i)})^{y^{(i)}} (1-h_\theta(x^{(i)}))^{1-y^{(i)}} = \sum_{i = 1}^M \log h_\theta(x^{(i)})^{y^{(i)}} + \log (1-h_\theta(x^{(i)}))^{1-y^{(i)}}\\
\log \mathcal{L}(\theta) = \sum_{i = 1}^M {y^{(i)}} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))
$$

Choose $\theta$ to **maximixe** $l(\theta)$. We'll do that by using **Batch gradient ascent**.

$$
\theta_j \, ;= \theta_j + \alpha \frac{\partial}{\partial \theta_j}l(\theta)
$$

By computing the partial derivatives, you get the following:

$$
\theta_j \, ;= \theta_j + \alpha \sum_{i = 1}^M (y^{(i)} - h_\theta (x^{(i)}))x_j^{(i)}
$$

Note that this equation is the same as the one from linear regression (by working out the negative signs, etc). What changed, however, is the definition of $h_\theta(x)$.

## Newton's Method

A much faster algorithm than gradient ascent to optmize $\theta$.

Consider the following problem: We have $f$, we want to find $\theta$ such that $f(\theta) = 0$.

We want to maximize $l(\theta)$, meaning we want to set the derivative $l'(\theta) = 0$

Newton's method takes an initial point and uses the tangent line to get the estimate value you want for the first iteration. Then, this estimate point is used to draw a new tangent line, getting a second (and more accurate) estimate.

### Mathematical view

On a graph of $f$ as a function of $\theta$, define $\Delta$ as the distance between the initial and the second $\theta$ values in Newton's method. In general, $\theta$ is obtained by subtracting $\Delta$ from the previous $\theta$. For the first two points, we have that $\theta^{(1)} = \theta^{(0)} -\Delta$

The derivative of $f$ can be written as the slope of the tangent line we drew, that is, $f'(\theta^{(0)}) = \frac{f(\theta^{(0)})}{\Delta}$. Therefore, $\Delta =\frac{f(\theta^{(0)})}{f'(\theta^{(0)})}$. This gives:

$$
\theta^{(t+1)} \,;= \theta^{(t)} - \frac{f(\theta^{(t)})}{f'(\theta^{(t)})}
$$

Plugging $l'$ back in, we arrive at the final formula for Newton's method:

$$
\theta^{(t+1)} \,;= \theta^{(t)} - \frac{l'(\theta^{(t)})}{l''(\theta^{(t)})}
$$

Newton's Method operates under **quadratic convergence**, which means the amount of significant figures of the result gets doubled by each iteation. For that reason, this method requires relatively few iterations to get to a satisfying result.

### Generalization: What if $\theta$ is a vector?

The assigment becomes the following:

$$
\theta^{(t+1)} ;= \theta^{(t)} + H^{-1}\nabla_\theta\, l
$$

Where $H$ is the Hessian matrix, defined as follows:

$$
H_{ij} = \frac{\partial^2\, l}{\partial \theta_i \,\partial \theta_j}
$$

For high dimensional parameters, Newton's Method becomes expensive due to the need to invert a large square matrix and compute several partial derivatives. For such problems, gradient descent is often a better choice.