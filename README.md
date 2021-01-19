<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

\documentclass{article}
\usepackage[letterpaper,top=0.25in,bottom=0.25in,left=0.5in,right=0.5in,includeheadfoot]{geometry}
\usepackage{amssymb,amsmath}
\usepackage{amsthm}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{tabu}
\usepackage{enumerate}
\usepackage{graphicx}
\usepackage{titlesec}
\usepackage{braket}
\usepackage{tikz}
\usetikzlibrary{automata,positioning,calc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{clrscode3e}
\RequirePackage{graphics}
\pagenumbering{gobble}
\usepackage{hyperref}
\title{Expected GINI Github Readme}
\author{Peter Gregory Husisian}
\date{January 2021}

\newcommand{\expectation}[1]{\mathbb{E}\left(#1\right)}
\DeclareMathOperator*{\argmax}{arg\,max}

\begin{document}

\maketitle

\section*{Background: Collection GINI Impurity}

The GINI impurity metric is a means of evaluating the homogeneity of a collection of objects, $Y = [y_1,\hdots,y_n]$, that each take on a value in the set $\{1,\hdots,J\}$. It is 0, its minimum, when $y_1 = \hdots = y_n$, and is at its maximum when $Y$ contains equal occurrences of each element of $\{1,\hdots,J\}$.

Explicitly, the GINI impurity of $Y$ is the probability that, two elements are sampled from $Y$ without replacement are not the same value. Let $p_k$ be the proportion of elements in $Y$ that have value $k$, then the GINI impurity of $Y$ is defined as:

\begin{align*}
    G_Y &= \sum_{k = 1}^{J} \overbrace{p_k}^{(1)} \overbrace{(1 - p_k)}^{(2)}
\end{align*}

\noindent
Where term (1) represents the probability of sampling value $k$ from $Y$, and term (2) represents the probability of sampling a value other than $k$ from $Y$. Simplifying the above, we get:

\begin{align*}
    G_Y &= \sum_{k = 1}^{J} p_k (1 - p_k) = \sum_{k = 1}^{J} p_k - p_k^2 = 
    \overbrace{\sum_{k = 1}^{J} p_k}^{=1} - \sum_{k = 1}^{J}p_k^2 = 1 - \sum_{k = 1}^{J}p_k^2
\end{align*}

\noindent
Let $1\{cond\}$ be 1 if $cond$ is true, 0 otherwise. We use this indicator notation to define $p_k = \frac{\sum_{i = 1}^{n} 1\{y_i = k\}}{n}$. Plugging this into the above, we get:

\begin{align*}
    G_Y &= 1 - \sum_{k = 1}^{J} \left(\frac{\sum_{i = 1}^{n} 1\{y_i = k\}}{n}\right)^2
    =
    1 - \frac{1}{n^2}\sum_{k = 1}^{J} \left(\sum_{i = 1}^{n} 1\{y_i = k\}\right)^2
\end{align*}

\section*{Background: Classification Impurity}

Consider a dataset $X = \{x_1,\hdots,n\}$ with corresponding labels $y_1,\hdots,y_n \in \{1,\hdots,J\}$. Let $f: X \rightarrow \{1,\hdots,B\}$ be an ``assigner-function'' which maps an input, $x_i$, to a corresponding collection, $f(x_i)$. We treat the impurity of the assigner-function $f$ as the average impurity the corresponding labels of every $x_i$ mapped to each collection, $\{1,\hdots,B\}$, weighted by the size of each collection. That is:

\begin{align*}
    G(f) &= \sum_{s = 1}^{B} \frac{\left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)G_{[y_i | f(x_i) = s]}}{n} \\
    %
    &=\sum_{s = 1}^{B} \frac{\left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)}{n} 
    \left(
        1 - \frac{
            \sum_{k = 1}^{J} \left(\sum_{i| f(x_i) = s} 1\{y_i = k\}\right)^2
            }
            {
            \left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)^2
            }
    \right)\\
    %
    &=
    \sum_{s = 1}^{B}
    \left(
         \frac{\left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)}{n}  
         - 
         \frac{
            \left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)\sum_{k = 1}^{J} \left(\sum_{i| f(x_i) = s} 1\{y_i = k\}\right)^2
            }
            {
            n \left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)^2
            }
    \right)\\
    %
    &=
    \sum_{s = 1}^{B}
    \left(
         \frac{\left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)}{n}  
         - 
         \frac{
            \sum_{k = 1}^{J} \left(\sum_{i| f(x_i) = s} 1\{y_i = k\}\right)^2
            }
            {
            n \sum_{i = 1}^{n} 1\{f(x_i) = s\}
            }
    \right)\\
    %
    &=
    \frac{1}{n}\overbrace{\sum_{s = 1}^{B}\left(\sum_{i = 1}^{n} 1\{f(x_i) = s\}\right)}^{=n}  
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i| f(x_i) = s} 1\{y_i = k\}\right)^2
        }
        {
        \sum_{i = 1}^{n} 1\{f(x_i) = s\}
        }\\
    %
    &= 
    1
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i| f(x_i) = s} 1\{y_i = k\}\right)^2
        }
        {
        \sum_{i = 1}^{n} 1\{f(x_i) = s\}
        }\\
    %
    &=
    1
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i = 1}^{n} 1\{f(x_i) = s\} 1\{y_i = k\}\right)^2
        }
        {
        \sum_{i = 1}^{n} 1\{f(x_i) = s\}
        }\\
    %
    &= 
    1
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i | y_i = k} 1\{f(x_i) = s\}\right)^2
        }
        {
        \sum_{i = 1}^{n} 1\{f(x_i) = s\}
        }
\end{align*}

\noindent
Thus, the GINI impurity of an assigner-function $f$ is:

\begin{align*}
    G(f) &= 1
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i | y_i = k} 1\{f(x_i) = s\}\right)^2
        }
        {
        \sum_{i = 1}^{n} 1\{f(x_i) = s\}
        }
\end{align*}

\section*{Stochastic Assignment GINI Impurity}

Now, suppose our assigner-function $f$ is a random function, and assigns point $x_i$ to collection $s$ with probability $p(s | x_i)$. In this instance, we define the ``Expected GINI'', $\expectation{G(f)}$, as the average expected impurity of the corresponding labels of every $x_i$ mapped to each collection, $\{1,\hdots,B\}$, weighted by the expected size of each collection. Each expectation is taken over all possible collections to which each $x_i$ may be assigned.

\begin{align*}
    \expectation{G(f)} &= 
    1
    -
    \frac{1}{n}\sum_{s = 1}^{B}
     \frac{
        \sum_{k = 1}^{J} \left(\sum_{i | y_i = k} p(s|x_i)\right)^2
        }
        {
        \sum_{i = 1}^{n} p(s|x_i)
        }
\end{align*}

\noindent
While this loss metric is loosely referred to as the ``expected GINI'', note that it is not the true expectation of the GINI impurity, as we are simply taking the expectation of the numerator and denominator of each summand. In practice, it still functions as desired, and is considerably less complex.


\section*{Expected GINI Gradient Derivation}

Let $\theta$ be some parameter of $p(s|x)$. Using the product rule, we get the following gradient for $\expectation{G(f)}$:

\begin{align*}
    \nabla_\theta \expectation{G(f)} &= 
    -
    \frac{1}{n}\sum_{s = 1}^{B}
    v_s \nabla_\theta u_s + u_s \nabla_\theta v_s
\end{align*}

\noindent
Where:


\begin{align*}
    u_s &= \frac{1}{\sum_{i = 1}^{n} p(s|x_i)}\\
    \nabla_\theta u_s &= - \frac{\sum_{i = 1}^{n} \nabla_\theta p(s|x_i)}{\left(\sum_{i = 1}^{n} p(s|x_i)\right)^2}\\
    v_s &= \sum_{k = 1}^{J} \left(\sum_{i | y_i = k} p(s|x_i)\right)^2\\
    \nabla_\theta v_s &= 2\sum_{k = 1}^{J} \left(\sum_{i | y_i = k} p(s|x_i)\right) \sum_{i | y_i = k} \nabla_\theta p(s|x_i)
\end{align*}

\section*{Tree-Based $p(s|x)$}

GINI impurity is a common metric used to train decision trees. Decision trees are grained greedily as minimizing the impurity of the leaves of the tree is a combinatoric optimization problem that quickly grows infeasible as tree depth increases (each node has $d \times n$ possible splits, where $d$ is the length of each observation).

Consider instead treating a decision tree as a stochastic process, in which an input $x$ traverses the tree by choosing sub-trees at random according to some probability distribution until a leaf is reached. If $s$ is a node of the tree, then let $s'$ denote the parent node of $s$. Then, the probability of $x$ reaching node $s$ is the probability of $x$ reaching its parent, $s'$, and then choosing the sub-tree rooted by $s$. 

Let $f_{s'}(s | x)$, henceforth called a ``split function'', be a probability distribution that gives the probability that an $x$ at node $s'$ chooses to traverse the sub-tree rooted at child node $s$. Then, for any node $s$, $p(s | x)$ is recursively defined as:

\begin{align*}
    p(\text{root} | x) &= 1\\
    p(s | x) &= f_{s'}(s | x)p(s' | x)
\end{align*}


\noindent
For sake of deriving the gradient, it will be useful to unravel the recursion above. Let $s^{(j)}$ refer to the $j$th level parent of $s$ -- for example, $s^{(1)} = s'$, $s^{(2)} = s''$, etc. Suppose node $s$ is of depth $d$, meaning $s^{(d)} = \text{root}$. Then:

\begin{align*}
    p(s | x) &= f_{s'}(s | x)p(s' | x)\\
    %
    &= f_{s'}(s | x) f_{s''}(s' | x) p(s'' | x)\\
    %
    &= \vdots\\
    %
    &= \prod_{i = 0}^{d-1} f_{s^{(i+1)}}(s^{(i)} | x)
\end{align*}



\noindent
Now, let $\theta_q$ refer to some parameter of $f_{q}$ (and only of $f_{q}$). Note that, if there does not exist a path from $q$ to $s$, then $f_{q}$ does not appear in the product above, so $\nabla_{\theta_q}p(s|x) = 0$. If such a path does exist, this means that there exists a $j$ such that $s^{(j)} = q$. $f_{q}$ appears exactly once in the above, so we get the following gradient:

\begin{align*}
    \nabla_{\theta_q} p(s|x) &= 
    \overbrace{\left(\prod_{i = 0}^{j-2} f_{s^{(i+1)}}(s^{(i)} | x)\right) \left(\prod_{i = j}^{d-1} f_{s^{(i+1)}}(s^{(i)} | x)\right)}^{= \frac{p(s|x)}{f_{q}(s^{(j-1)}|x)}} \nabla_{\theta_q}f_{q}(s^{(j-1)} | x)\\
    %
    &= \frac{p(s|x)}{f_{q}(s^{(j-1)}|x)} \nabla_{\theta_q}f_{q}(s^{(j-1)} | x)
\end{align*}



\section*{Minimizing Expected GINI of Tree-Based $p(s | x)$}

Now that we have the defined the gradient of a tree-based $p(s|x)$, we can simply plug it in to the gradient of $\expectation{G(f)}$. However, we sum only over the leaves of the tree, as the leaf a given $x$ falls in is consider the collection to which $x$ is assigned.

\section*{Choice of $f_{s'}(s|x)$}

So long as $f_{s'}(s|x)$ is a valid probability distribution over the possible children of $s'$, any function may be chosen. For example, $f_{s'}$ may be represented as a logistic regression model, kernel logistic-regression, or even a neural network, although one would almost certainly be better served by simply applying more complex methods directly to the problem than to leverage complex split functions.


TODO: attach logistic regression tree and non-linear split tree pictures



\section*{Classification with Tree-Based $p(s|x)$}

Let $\bar{f}(x)$ assign a leaf node to point $x$ by having $x$ traverse the tree by always taking the most likely child according to each split function. The most straight-forward method of assigning a class to a point $x$ is to classify it as the most frequently occurring label of the training data that falls into leaf $\bar{f}(x)$ according to $\bar{f}$, just as a decision tree does. This is a desirable property, as the most frequently occurring label can be pre-computed, so calculating the predicted class of $\bar{f}(x)$ can be done in $O(d)$ time, where $d$ is the depth of the tree. This is significantly faster than other tree-based methods, which require a weighted sum over all nodes of the tree, and has the added effect of "divvying up" the input space into piecewise-constant regions, just as traditional decision trees do.


%argmax method
%proportions method where each leaf is weighted by probability (probably the best)
%stochastic method where x takes a random path
%argmax child (very fast, mimics a traditional decision tree exactly)


\section*{Computational Concerns of Tree-Based $p(s | x)$}

Let $d$ represent the depth of the tree, which we will assume to be binary (although this need not be the case, since $f_{s'}(s | x)$ may be chosen arbitrarily). Then, a naive calculation of $p(s | x)$ for each leaf $s$ takes $O(d2^d)$ time. However, $p(s | x)$ could be computed for each leaf via depth-first search, which guarantees nodes are visited in an order such that all prerequisite calculations have already been completed, which takes $O(2^d)$ time, and also has the advantage of computing, and storing, $p(s|x)$ for every node $s$ in the tree (not just leaves), if this is desired. 


In order to compute the gradient $\nabla_{\theta_q} p(s|x)$ for all nodes $q$ and leaves $s$, we compute $p(s|x)$ for all leaves $s$ using the method described above in $O(2^d)$ time. Each leaf $s$ has at most $d$ possible $q$ such that $\nabla_{\theta_q}p(s|x) \neq 0$, as these $q$ are all the nodes reachable by traversing the tree in reverse starting from $s$. Thus, assuming each $\nabla_{\theta_q}f(s^{(j-1)} | x)$ and $f_{\theta_q}(s^{(j-1)} | x)$ take constant-time to compute, the non-zero gradients $\nabla_{\theta_q}p(s|x)$ can be computed in $O(2^d + 2^{d-1}d) = O(2^{d}d)$ time.

If faster computation is required, an unbiased estimation of the gradient is calculated as follows:

\begin{enumerate}[1)]
    \item Sample a stochastic path down the tree for $x$ to take by sampling each random child according to their split probability. Let leaf $s$ refer to the leaf into which $x$ falls.
    \item Compute only $p(s|x)$ in $O(d)$ time (better yet, compute it while forming the stochastic path by multiplying the associated split probabilities).
    \item For all $q$ such that a path exists to $s$, of which there are at most $d$, compute $\nabla_{\theta_q}p(s|x)$. Set the gradients for all other $q$ and all other non-$s$ leaves to 0 (that is, simply don't iterate over them).
\end{enumerate}

\noindent
This calculation requires only $O(d)$ time, and is equal to the exact full gradient of all leaves of the tree with respect to all parameters of the tree in the expectation. For particularly large trees, it may be a wise choice to increase training iterations and use this unbiased estimator. This approach is akin to the stochastic gradient descent algorithm. If a ``mini-batch'' estimate is desired, one could sample $B$ different stochastic paths and compute the gradients of these paths using the same technique described above, zeroing out all leaves and parameters that do not correspond to any of the $B$ sampled paths, which takes $O(Bd)$ time. Similar to batch gradient descent, this approach is also equal in the expectation to the true gradient, but has a smaller variance.


Through observation of the training process, I noted that split parameters near the bottom of the tree varied wildly during the first iterations. This is likely explained by the fact that the higher-up splits in the tree were nowhere near convergence, which would significantly perturb the expected set of inputs of each node. This was especially an issue when gradients were approximated as described above, as lower splits would witness drastically different sets of data every iteration while the higher splits still converged. I experimented with dynamically pruning and growing the tree to mitigate this issue, and it decreased convergence time notably. 


\section*{Advantages and Quirks of Tree-Based $p(s|x)$}

This decision tree is globally optimized using calculus-based methods such as Gradient Descent, as opposed to traditional decision trees, which are greedily optimized discretely, and as a consequence, are forced to have axis-aligned splits. The convergence properties of the Expected GINI applied to a tree-based $p(s|x)$ have not yet been explored, but there is a possibility that convergence to a global optima may be guaranteed. Note that multiple equivalent global optima must exist, as one could simply swap the children of any node, and modify its split function accordingly, to yield a tree with the exact same expected impurity.

As the tree can have more leaves than there are labels, it is possible to have multiple different leaves correspond to the same class. This may be desirable if points of the same label occupy non-contiguous regions of space, as these could be ``teased apart'' and assigned to separate leaves of the tree. Even trees of fairly low depth can accomplish this effect, and the extent to which this occurs depends on the number of classes of the problem, and the number of leaves of the tree, which is on the order of $O(2^{d})$.


The choice of arbitrary split function results in loss of tree interpretability, which is a desirable trait of traditional decision trees, as they are axis-aligned, meaning that for any classification, one could express the set of logical conditions that were satisfied that lead to that classification. One possible way to regain the interpretability of traditional decision trees using this globally optimized context is to use linear split functions, and add the following regularization term to the expected GINI:

\begin{align*}
    \lambda \sum_{\forall q} (\theta_q^T \theta_q - \|\theta_q\|_\infty^2)
\end{align*}

\noindent
Which equals 0, its minimum, if and only if each $\theta_q$ contains exactly one non-zero element, which result in axis-aligned splits. A regularizer simular to that of the Lasso criterion is also promising, but it is significantly more complex to implement.


Further, the tree architecture could be modified such that, as a point passes through a node, a non-linear transformation is applied, resulting in a ``neural network tree''. I have derived the math behind this, but it is incredibly complex, and requires a special form of backpropagation to be made remotely efficient. Neural network architectures are already prone to overfitting, and I feel this would be even moreso the case in tree form, so I doubt this is a very practical application, albeit cool.


\end{document}






