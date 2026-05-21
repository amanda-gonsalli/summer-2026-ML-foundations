# Supervised Learning: Linear Regression and Gradient Descent 

You have a training set $(x,y)$ fed into a learning algorithm. It generates a function called hypothesis. This function is used to estimate the output for a test inout (an x-value for which we don't know the y-value).

## How to represent the hypothesis h?
You may have more than one input features, like $x_1$, $x_2$, $x_3$, etc.

$$ 
h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

Or, more concisely:

$$
h(x) = \sum_{j = 0}^2 \theta_j x_j
$$

And we define $x_0 = 1$

Therefore, we have the parameter $\theta$ defined as

$$
\theta =
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
... \\
\theta_n
\end{bmatrix}
$$

and $x$ the input features

$$
x = 
\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
... \\
x_n
\end{bmatrix}
$$

Define $M$ as the number of training examples, that is, the number of datapairs $(x,y)$
Define $y$ as the output/ target variable
Define $(x^{(i)}, y^{(i)})$ as the i'th training example. In that case, $i$ runs from 1 to M
Finally, define $n$ as the number of features. In the example above, n = 2

With that in mind, if we have $n$ features, $x$ and $\theta$ are $n+1$ dimensional vectors

## How do you choose parameters $\theta$?

Choose $\theta$ such that $h(x) \approx y$ for the training examples

Note that instead of writing $h(x)$, we write $h_{\theta} (x)$ to emphasize that the hypothesis depends on both the parameter $\theta$ and the input $x$

In linear regression, we want to minimize the square different of what the hypothesis outputs and the actual output from the training set. We do that by choosing the optimal $\theta$

Define the cross-function $J(\theta)$ as follows:
$$
J(\theta) = \frac{1}{2}\sum_{i = 1}^M(h_\theta(x^{(i)}) - y)^2
$$

We want to minimize $J(\theta)$

## Gradient Descent

Start with some $\theta$, say $\theta = \vec{0}$ (this is a vector with all zeroes in it)

Keep changing $\theta$ to reduce $J(\theta)$

### Intuition
In a 3d graph, you can plot $\theta_0$ and $\theta_1$, mapping them to the $J(\theta)$ value obtained, we want values for $\theta_0$ and $\theta_1$ that make the height of this surface the smallest possible.

We start at some point in the surface (initial parameter values). Look around that point and see which direction will bring you to the lowest possible value in one tiny step. Then, we repeat this process on the new location. 

Eventually, you'll get to a point where there's no direction that brings you further down. This is a minimum of the the cross-function. However, there can be many local minima.

### Formal definition
$$
\theta_j \, ;= \theta_j -  \alpha \frac{\partial}{\partial \theta_j} J(\theta) \hspace{10pt} (j = 0, 1, 2, ..., n) 
$$

Note: $;=$ denotes that the value on the right is being assigned to the value on the left

In the equation, $\alpha$ is the learning rate, and that gets multiplied by the direction of steepest ascent (gradient of the cross-function)

Explicitly writing the derivative, we have
$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \frac{1}{2}(h_\theta (x) - y)^2 = (h_\theta (x) - y) \, \frac{\partial}{\partial \theta_j} (h_\theta (x) - y) \\
 = (h_\theta (x) - y) \, \frac{\partial}{\partial \theta_j} (\theta_0 x_0 + \theta_1 x_1 + \dots + \theta_n x_n)\\
 = (h_\theta (x) - y) \, \frac{\partial}{\partial \theta_j} (\theta_j x_j) \\
\frac{\partial}{\partial \theta_j} J(\theta)  = (h_\theta (x) - y) \, x_j
$$

Now remember there are $M$ training examples, so the equation becomes the following:

$$
\theta_j \, ;= \theta_j - \alpha \, \sum_{i=1}^M(h_\theta (x^{(i)}) - y^{(i)}) \, x^{(i)}_j \hspace{10pt} (j = 0, 1, 2, ..., n) 
$$

Gradient descent is about repeating this iteration until it converges.

Graphically, $J(\theta)$ is a degree 2 expression, so the graph is the 3d extension of a parabola. It therefore has no local minima other than the global minimum. Also, the countour map of the cross-function is composed of ellipses. This ensures the algorithm will eventually converge to the gloal minimum.

### How to choose $\alpha$?

* Too big: overshoot ($J(\theta)$ increases)
* Too low: takes too many iterations and computational power

Good strategy: try multiple values on an exponential scale (0.01, 0.02, 0.04 etc)

Sometimes, this whole process is called **Batch gradient descent**. It means the algorithm iterates over the whole dataset in order to make each step of the gradient descent. In large datasets, this becomes very slow and expensive.

### An alternative to batch gradient descent: Stochastic gradient descent

Repeat the following:

For $i = 1$ to $M$:

$\hspace{10pt} \theta_j \, ;= \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})\, x^{(i)}_j$

Update for every $j$.

Note: instead of iterating over all $i$ for one update (one step in the descent), this method considers only one $i$, updates $\theta_j$, then move on to the next $i$. 

You change the parameter based on one example at a time, which makes the steps not go perpendicular to the countour map lines (takes "longer" paths). But for larger datasets this method makes faster progress. 

Stochastic gradient doesn't converge exactly to the global minimum, but still gets very good parameter values. What is often done is decreasing the learning rate on this method to get a better estimate, but that still won't be as precise as Batch. Stochastic is still preferred for large datasets due to Batch being extremely slow for those cases.

## The Normal Equation (only works for Linear Regression)

Define $\nabla_\theta J(\theta)$, where $\theta \in \mathbb{R}^{n+1}$ as follows:

$$
\nabla_\theta J(\theta) = 
\begin{bmatrix}
\frac{\partial}{\partial \theta_0} J \\
\frac{\partial}{\partial \theta_1} J \\
\frac{\partial}{\partial \theta_2} J
\end{bmatrix}
$$

Suppose we have the matrix $A \in \mathbb{R}^{2x2}$, and some function $f: \mathbb{R}^{2x2} \rightarrow \mathbb{R}$.

Define the matrix derivatvie:
$$
\nabla_A f(A)=
\begin{bmatrix}
\frac{\partial}{\partial A_{11}}f & \frac{\partial}{\partial A_{12}}f \\
\frac{\partial}{\partial A_{21}}f & \frac{\partial}{\partial A_{22}}f
\end{bmatrix}
$$

We want to minimize the cross-function, that is, $\nabla_\theta J(\theta) \stackrel{\text{set}}{=} \vec{0}$

If $A$ is a square matrix, define the "trace of A" $tr A = \sum_i A_{ii}$ as the sum of diagonal entries. 

Some properies of the "trace":
* $tr A = tr A^T$
* Define $f(A) = tr AB$. Then, $\nabla_A f(A) = B^T$
* $tr AB = tr BA$
* $tr ABC = tr CAB$
* $\nabla_A \, tr AA^TC = CA + C^TA$
    * Think of this one as analogous to $\frac{d}{da}a^2c = 2ac$

Now, take the cross-function again

$$
J(\theta) = \frac{1}{2}\sum_{i = 1}^M (h(x^{(i)}) - y^{(i)})^2
$$

Define the design matrix $X$ as follows:

$$
X = 
\begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\dots \\
(x^{(M)})^T \\
\end{bmatrix}
$$

Then, 
$$
X\theta = 
\begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\dots \\
(x^{(M)})^T \\
\end{bmatrix} 

\begin{bmatrix}
\theta_0\\
\theta_1 \\
\dots \\
\theta_M
\end{bmatrix}
= 
\begin{bmatrix}
(x^{(1)})^T \theta \\
(x^{(2)})^T \theta \\
\dots \\
(x^{(M)})^T \theta 
\end{bmatrix} 
= 
\begin{bmatrix}
h_\theta(x^{(1)}) \\
h_\theta(x^{(2)}) \\
\dots \\
h_\theta(x^{(M)}) \\
\end{bmatrix}
$$

Now define a vector $\vec{y}$ such that it stacks all the example outputs in a column vector:

$$
\vec{y} = 
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\dots \\
y^{(M)}
\end{bmatrix}
$$

Then, 
$$
J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y})
$$

Explicitly, the first term is:
$$
X\theta - \vec{y} = 
\begin{bmatrix}
h_\theta(x^{(1)}) \\
h_\theta(x^{(2)}) \\
\dots \\
h_\theta(x^{(M)})
\end{bmatrix}
-
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\dots \\
y^{(M)}
\end{bmatrix}
= 
\begin{bmatrix}
h_\theta(x^{(1)}) - y^{(1)} \\
h_\theta(x^{(2)}) - y^{(2)}\\
\dots \\
h_\theta(x^{(M)}) - y^{(M)}
\end{bmatrix}
$$

This is just all the errors from the algorithm.

Since for any matrix $Z$ we have that $ZZ^T = \sum_i Z^2$, we expression for the cross-function returns to the one we previously had. 

$$
J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2}\sum_{i = 1}^M (h(x^{(i)}) - y^{(i)})^2
$$

Finally, consider the derivative of the cross-function with respect to $\theta$

$$
\nabla_\theta J(\theta) = \nabla_\theta \, \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2} \nabla_\theta \,(\theta^TX^T - \vec{y}^T)(X\theta - \vec{y}) \\
= \frac{1}{2} \nabla_\theta \, (\theta^TX^TX\theta - \theta^T X^T\vec{y} - \vec{y}^TX\theta + \vec{y}^T\vec{y}) = \frac{1}{2}(X^TX\theta + X^TX\theta - X^T\vec{y}-X^T\vec{y})\\
\nabla_\theta J(\theta) = X^TX\theta - X^T\vec{y}
$$

Now we set it to zero

$$
\nabla_\theta J(\theta) = X^TX\theta - X^T\vec{y} = 0\\
X^TX\theta = X^T\vec{y}
$$

This is the **normal equation**. And the optimal value for the parameter is
$$
\theta = (X^TX)^{-1}X^T\vec{y}
$$