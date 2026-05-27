# Perceptron & Generalized Linear Model
Last lecture we saw the "sigmoid" function for classification problems. However, this is not the only option.

## Perceptron

$$
g(z) = 
\begin{cases}
1, \, z \geq 0\\
0, \, z < 0
\end{cases}
$$

With this function, the hypothesis becomes:

$$
h_\theta(x) = g(\theta^Tx)
$$

And the update rule:

$$
\theta_j \, ;= \theta_j + \alpha \, (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}_j
$$

But note that what changes is the hypothesis $h_\theta(x)$ function definition.

The quantity $y^{(i)} - h_\theta(x^{(i)})$ is a scalar. It can be:

* 0, if the algorithm got it right (meaning both $y$ and $h(x)$ are 0 or 1)

* $\pm 1$, is the algorithm got it wrong:

    * 1 if $y = 1$ and $h(x) = 0$
    * -1 if $y = 0$ and $h(x) = 1$

Suppose the input space filled with squares and circles, we want the algorithm to separate them. It then creates a bounday line in the input space in which $\theta^Tx = 0$. The vector normal to that line is the parameter $\theta$. Now, if the algorithm gets a new training example and misclassifies it, then it will follow the parameter update rule. The quantity $y^{(i)} - h_\theta(x^{(i)})$ will be either 1 or -1, and so we'll have $\theta_J \, ;= \theta_j \pm \alpha x^{(i)}_j$. This is change the direction of the vector $\theta$ normal to the line, effectively rotating the line in the input space, as so to separate the two different catergories correctly.

## Exponential Families

They're a class of probability dirstributions.

A exponential family is one whose PDF (probability density function) can be written as:

$$
P(y; \eta) = b(y)\, \exp[\eta^T T(y) - a(\eta)]
$$

Where:
* $y$ is the data
* $\eta$ is the natural parameter
* $T(y)$ is called sufficient statistic. Here, we'll always have $T(y) = y$
* $b(y)$ is called base measure
* $a(\eta)$ is called the log-partition function. This is a normalizing parameter (so that the pdf integrates to 1)

Note that $\eta$ and $T(y)$ must match dimensions, while the rest are scalars (except $y$ which can be a vector too).

### Bernoulli Distribution (Models Binary Data)
* $\phi$ is the probability of the event happening or not
* $p(y;\phi) = \phi^y(1-\phi)^{(1-y)}$

We want this pdf to be of the form of the exponential family pdf's.

$$
p(y'\phi) = \exp [\log (\phi^y(1-\phi)^{(1-y)})]\\
= \exp [\log\left( \frac{\phi}{1 -\phi}\right)y + \log(1-\phi)]
$$

From this form, we can match components and get:

* $b(y) = 1$
* $T(y) = y$
* $\eta = \log\left(\frac{\phi}{1-\phi}\right) \Rightarrow \phi = \frac{1}{1 + e^{-\eta}}$
* $a(\eta) = -\log(1-\phi) = -\log\left(1 - \frac{1}{1 + e^{-\eta}}\right) = -\log \left(\frac{1}{1 + e^{\eta}}\right) = \log(1 + e^{\eta})$

Therefore the Bernoulli is a member of the exponential family.

### Gaussian Distribution (with fixed variance)

Assume $\sigma^2 = 1$

* $p(y;\mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y - \mu)^2}{2}\right)$

Rewrite it:

$$
p(y;\mu) = \frac{1}{\sqrt{2\pi}}\, e^{-\frac{y^2}{2}} \exp\left(\mu y - \frac{1}{2} \mu^2\right)
$$

* $b(y) = \frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}$
* $T(y) = y$
* $\eta = \mu$
* $a(\eta) = \frac{1}{2}\mu^2 = \frac{1}{2} \eta^2$

### Why do we use the Exponential Family?

It has some nice properties:

* MLE (maximum likelihood estimation) with respect to $\eta$ is concave
    * NLL (negtive log likelihood - basically the minimizing cost function) with respect to $\eta$ convex
* $E[y;\eta] = \frac{\partial}{\partial \eta} a(\eta)$ is the mean of the distribution
* $Var[y;\eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$ is the variance of the distribution

This means we can get the mean and the variance without integration, which is otherwise needed.

## GLM: Generalized Linear Model

Assumptions:

* $y|x;\theta \sim$ exponential family of $\eta$ ($\sim$ is a member of)
* $\eta = \theta^T x$
* At test time: output will be $E[y|x;\theta]$
    * Given $x$, the mean of the distribution is the output of the function
    * $h_\theta(x) =E[y|x;\theta] $
* At training time: MLE on $\log p(y^{(i)};\theta^Tx^{(i)})$
* Once you get the optimal $\theta^Tx$ value, you reparametrized in terms of $\eta$ to learn the parameters from the exponential family (b, a, T, etc)

### GLM Training

For any distribution you choose, the learning update rule is the same as before:

$$
\theta_j \, ;=\theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j
$$

Some terminology:
* $\eta$ - natural parameter is linked to the mean of distribution by the canonical response function $\mu = E[y;\eta] = g(\eta) = \frac{\partial}{\partial \eta}a(\eta)$
* $\eta = g^{-1}(\mu)$ is the canonical link function

We have 3 parametrizations:
1. Model Parameters $\theta$
2. Natural Parameters $\eta$ for the exponential family
3. Canonical Parameters for the distribution

    a. $\phi$ for Bernoulli

    b. $\mu$, $\sigma^2$ for Gaussian

Learn $\theta$ through gradient ascent, use the choice $\eta = \theta^Tx$ to get the normal parameter, then use $g$ and $g^{-1}$ to get the canonical parameter.

For instance, in logistic regression, the choice of distribution is a Bernoulli, so:
 
$$
h_\theta(x) = E[y|x;\theta] = \phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\theta^Tx}} 
$$

## Softmax Regression

* $k$ number of classes (square, triangle, circle, etc)
* $y$ is a matrix of all zeros except it has 1 on the position that indicates the class
* Each class has its own parameter $\theta_{class}$

Given an example $x$, we generate a plot of vertical bars $\theta^Tx$ over the classes. This is called the logit space. We then exponentiate it (making all values positive) and normalize it by dividing everything by the sum of all of them. This way, we get a probablity distribution (pdf), where the sum of all the bars give 1. Call this the hypothesis function $\hat{p}(y)$.

The actual output $p(y)$ can be thought of as a plot of vertical bars with all classes 0 except the correct one (evaluated to 1).

The learning algorithm we build will make the first plot similar to the second plot (make $\hat{p}(y)$ similar to $p(y)$). This is to minimize the cross-entropy of the two distributions.

$$
CrossEnt(p\hat{p}) = -\sum p(y)\log \hat{p}(y), \,\text{for $y$ in all classes}
$$

Since $p(y)$ is 1 for one of the classes and zero for all others, the expression becomes $-\log \hat{p}(y_{?})$, being "?" the correct class. By definition of the hypothesis function, this becomes:

$$
CrossEnt(p\hat{p}) = -\log \frac{e^{\theta_{?}^Tx}}{\sum_{classes} e^{\theta^Tx}}
$$

Then, do gradient descent with respect to the parameters for each class.