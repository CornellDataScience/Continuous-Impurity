# Continuous-Impurity
Goal: To define a continuous analogue to an impurity metric so that decision trees can be trained with calculus-trained split-planes that are not forced to be parallel to an axis. A continuous analogue of GINI impurity, defined as:


<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" title="G = 1 - \frac{1}{|X|} \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} 1\left \{ f(X_i) = k\right \})^2} {\sum_{i} 1\left \{ f(X_i)=k\right \}}" /></a>

was determined by taking it's expectation -- considering the placement of a given input, x, into a given subset, k -- as being probabilistic in nature, dictated by the conditional probability p(k|x). p(k|x) is used when calculating the expectation of each indicator variable, and is the model to be trained when minimizing the expected impurity.

<a href="https://www.codecogs.com/eqnedit.php?latex=E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" title="E(G) = 1 - \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} p(k|X_i))^2} {|X|\sum_{i} p(k|X_i)}" /></a>


For use in caculus-optimized models, the gradient of the expected GINI impurity with respect the parameters of the model must be calculated, allowing for an arbitrary model to be trained with expected GINI by replacing the definition of p(k|x) and its gradient with respect to its parameters, theta.


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;\textup{where}&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;\textup{where}&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" title="\newline \triangledown _\theta E(G) = \frac{-1}{|X|} \sum_{k}v_k(\triangledown _\theta u_k) + u_k (\triangledown _\theta v_k) \newline \textup{where} \newline u_k = \frac {1}{\sum_ i p(k|X_i)} \newline \triangledown _\theta u_k = \frac {-\sum _i \triangledown _\theta p(k|X_i)} {(\sum _i p(k|X_i))^2} \newline v_k = \sum _l (\sum _{i|y_i=l} p(k|X_i))^2 \newline \triangledown _\theta v_k = 2 \sum _l (\sum _{i|y_i = l} p(k|X_i)) \sum _{i|y_i = l} \triangledown _\theta p(k|X_i)" /></a>

----------------------------------------------------------------------------------------------------------------

So far, continuous impurity has been been applied to the following forms of p(k|x) and their associated required gradients:

**Logistic Regression-Style p(k|x)**:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;x}}&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;x})^{-1}}{\partial&space;({\theta^Tx})}&space;\triangledown_\theta&space;(\theta^Tx)&space;=&space;p(k|x)(1-p(k|x))x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;x}}&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;x})^{-1}}{\partial&space;({\theta^Tx})}&space;\triangledown_\theta&space;(\theta^Tx)&space;=&space;p(k|x)(1-p(k|x))x" title="p(k|x) = \frac {1}{1+e^{-\theta^T x}} \newline \triangledown _\theta p(k|x) = \frac {\partial (1+e^{-\theta^T x})^{-1}}{\partial ({\theta^Tx})} \triangledown_\theta (\theta^Tx) = p(k|x)(1-p(k|x))x" /></a>

Results: 


![Logistic 1](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Continuous%20impurity%20first%20working%20result.png?raw=true)


A beneficial trait of models trained with expected GINI is that they can be used as the "split" model of decision trees, allowing more expressive decision boundaries than what was previously possible. Some examples of using the Logistic Regression-style p(k|x) as the split function for 2-depth decision trees:

![Continuous Decision Tree 1](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Continuous%20Imuprity%20Decision%20Tree.png?raw=true)

![Continuous Decision Tree 2](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Continuous%20Imuprity%20Decision%20Tree%202.png?raw=true)

*(Note the non-jagged decision boundaries that a normal decision tree is forced to have)

---------------------------------------------------------------------------------------------------------------

**Nonlinear Pretransformation Logistic Regression-Style p(k|x)** (t(x) is a non-linear transformation of x created by applying a non-linear activation function, a, to each element of the product of Ax, where A is a matrix of transformation parameters):

<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;t(x)}}&space;\newline&space;\newline&space;\textup{where:}&space;\newline&space;\newline&space;t_0(x)&space;=&space;1,&space;\newline&space;t_k(x)&space;=&space;a((Ax)_k)&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;t(x)})^{-1}}{\partial&space;({\theta^T&space;t(x)})}&space;\triangledown_\theta&space;(\theta^T&space;t(x))&space;=&space;p(k|x)(1-p(k|x))t(x)&space;\newline&space;\frac&space;{\partial&space;p(k|x)}&space;{\partial&space;A_{kr}}&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;t(x)})^{-1}}{\partial&space;({\theta^T&space;t(x)})}&space;\frac&space;{\partial&space;\theta^T&space;t(x)}{\partial&space;t_k(x)}&space;\frac&space;{\partial&space;t_k(x)}&space;{\partial&space;A_{kr}}&space;\newline&space;\newline&space;\textup{where:}&space;\newline&space;\newline&space;\frac&space;{\partial&space;t_k(x)}&space;{\partial&space;A_{kr}}&space;=&space;\frac&space;{\partial&space;a(Ax)}{\partial&space;Ax}&space;\frac&space;{\partial&space;Ax}{\partial&space;A_{kr}}&space;=&space;x_r\frac&space;{\partial&space;a(p)}{\partial&space;p}|^{p=Ax}&space;\newline&space;\newline&space;\Leftrightarrow&space;\frac&space;{p(k|x)}{A_{kr}}&space;=&space;x_r&space;\theta_k&space;p(k|x)(1-p(k|x))\frac{\partial&space;a(p)}{\partial&space;p}|^{p=Ax}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;t(x)}}&space;\newline&space;\newline&space;\textup{where:}&space;\newline&space;\newline&space;t_0(x)&space;=&space;1,&space;\newline&space;t_k(x)&space;=&space;a((Ax)_k)&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;t(x)})^{-1}}{\partial&space;({\theta^T&space;t(x)})}&space;\triangledown_\theta&space;(\theta^T&space;t(x))&space;=&space;p(k|x)(1-p(k|x))t(x)&space;\newline&space;\frac&space;{\partial&space;p(k|x)}&space;{\partial&space;A_{kr}}&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;t(x)})^{-1}}{\partial&space;({\theta^T&space;t(x)})}&space;\frac&space;{\partial&space;\theta^T&space;t(x)}{\partial&space;t_k(x)}&space;\frac&space;{\partial&space;t_k(x)}&space;{\partial&space;A_{kr}}&space;\newline&space;\newline&space;\textup{where:}&space;\newline&space;\newline&space;\frac&space;{\partial&space;t_k(x)}&space;{\partial&space;A_{kr}}&space;=&space;\frac&space;{\partial&space;a(Ax)}{\partial&space;Ax}&space;\frac&space;{\partial&space;Ax}{\partial&space;A_{kr}}&space;=&space;x_r\frac&space;{\partial&space;a(p)}{\partial&space;p}|^{p=Ax}&space;\newline&space;\newline&space;\Leftrightarrow&space;\frac&space;{p(k|x)}{A_{kr}}&space;=&space;x_r&space;\theta_k&space;p(k|x)(1-p(k|x))\frac{\partial&space;a(p)}{\partial&space;p}|^{p=Ax}" title="\newline p(k|x) = \frac {1}{1+e^{-\theta^T t(x)}} \newline \newline \textup{where:} \newline \newline t_0(x) = 1, \newline t_k(x) = a((Ax)_k) \newline \triangledown _\theta p(k|x) = \frac {\partial (1+e^{-\theta^T t(x)})^{-1}}{\partial ({\theta^T t(x)})} \triangledown_\theta (\theta^T t(x)) = p(k|x)(1-p(k|x))t(x) \newline \frac {\partial p(k|x)} {\partial A_{kr}} = \frac {\partial (1+e^{-\theta^T t(x)})^{-1}}{\partial ({\theta^T t(x)})} \frac {\partial \theta^T t(x)}{\partial t_k(x)} \frac {\partial t_k(x)} {\partial A_{kr}} \newline \newline \textup{where:} \newline \newline \frac {\partial t_k(x)} {\partial A_{kr}} = \frac {\partial a(Ax)}{\partial Ax} \frac {\partial Ax}{\partial A_{kr}} = x_r\frac {\partial a(p)}{\partial p}|^{p=Ax} \newline \newline \Leftrightarrow \frac {p(k|x)}{A_{kr}} = x_r \theta_k p(k|x)(1-p(k|x))\frac{\partial a(p)}{\partial p}|^{p=Ax}" /></a>

Where x has a 1 appended to the end to allow for a transformation bias, and a 1 is prepended to the output of t(x) to allow for a classification bias.


With a = tanh, A with 4 rows: 


![Nonlinear Pretransform 1](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Neural-network-style%20pretransform%20continuous%20impurity.png?raw=true)


With a = tanh, A with 6 rows:


![Nonlinear Pretransform 2](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Neural-network-style%20pretransform%20continuous%20impurity%202.png?raw=true)


Using this model as a split function in a greedily trained decision tree often proves finnicky, as its cost is not globally convex, so optimal convergence is not guaranteed. If a node's model does not converge (usually this means that the model classifies all, or almost all, points as belonging to one subset), then the node is forced to become a leaf, as it would have one child through which no data flows during training, and another through which all data flows during training, effectively making that child identically trained to itself. In such cases, the subtree is forced to terminate training prematurely, leading to a bad fit.

---------------------------------------------------------------------------------------------------------------
**TODO: ACCIDENTALLY DEFINED THIS WRONG. REWRITE WITH CORRECTED REPLACEMENT**
**In progress: Globally Optimal, Non-Greedily Trained Decision Trees with Arbitrary Split Function**


Globally optimal decision trees should be able to be found by instead minimizing the expected GINI impurity of the leaves of the decision tree. Hopefully this will fix issues with using the Nonlinear Pretransformation Logistic Regression-Style p(k|x) as the split function for a greedily-trained decision tree. This is accomplished by the following definition of p(k|x), where p(k|x) is instead the probability of x *passing through* node k as it travels down the tree. Let l(k) be whether node k is a leaf, D(k) be the depth of node k, k' represent the parent node of k, and f_k(x) be the probability split function node N contains. Then:


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;p(k|x)&space;=&space;p(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;p(\textup{root}|x)&space;=&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;p(k|x)&space;=&space;f_k(x)p(k'|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;p(k|x)&space;=&space;p(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;p(\textup{root}|x)&space;=&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;p(k|x)&space;=&space;f_k(x)p(k'|x)" title="\newline \textup{Boundary Conditions:} \newline p(k|x) = p(k'|x) \textup{ if } l(k), \newline p(\textup{root}|x) = f_{\textup{root}}(x) \newline \newline \textup{Recursive Condition:} \newline p(k|x) = f_k(x)p(k'|x)" /></a>


Despite the compounded products that appear in the recursive definition, it's gradient surprisingly does not require compounded product rules to compute, since either or both product rule terms of the recursive condition turns out to be zero by case analysis:


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(k|x)&space;=&space;\triangledown_&space;{\theta&space;_q}&space;p(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(\textup{root}|x)&space;=&space;1\{q=\textup{root}\}&space;\triangledown_&space;{\theta&space;_q}&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(k|x)&space;=&space;\begin&space;{cases}&space;0&space;&&space;D(q)>D(k)&space;\\&space;p(k'|x)&space;\triangledown&space;_{\theta&space;_q}&space;f_k(x)&space;&&space;D(q)&space;=&space;D(k)&space;\\&space;f_k(x)&space;\triangledown&space;_{\theta&space;_q}&space;p(k'|x)&space;&&space;D(q)&space;<&space;D(k)&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(k|x)&space;=&space;\triangledown_&space;{\theta&space;_q}&space;p(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(\textup{root}|x)&space;=&space;1\{q=\textup{root}\}&space;\triangledown_&space;{\theta&space;_q}&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;p(k|x)&space;=&space;\begin&space;{cases}&space;0&space;&&space;D(q)>D(k)&space;\\&space;p(k'|x)&space;\triangledown&space;_{\theta&space;_q}&space;f_k(x)&space;&&space;D(q)&space;=&space;D(k)&space;\\&space;f_k(x)&space;\triangledown&space;_{\theta&space;_q}&space;p(k'|x)&space;&&space;D(q)&space;<&space;D(k)&space;\end{cases}" title="\newline \textup{Boundary Conditions:} \newline \triangledown_ {\theta _q} p(k|x) = \triangledown_ {\theta _q} p(k'|x) \textup{ if } l(k), \newline \triangledown_ {\theta _q} p(\textup{root}|x) = 1\{q=\textup{root}\} \triangledown_ {\theta _q} f_{\textup{root}}(x) \newline \newline \textup{Recursive Condition:} \newline \triangledown_ {\theta _q} p(k|x) = \begin {cases} 0 & D(q)>D(k) \\ p(k'|x) \triangledown _{\theta _q} f_k(x) & D(q) = D(k) \\ f_k(x) \triangledown _{\theta _q} p(k'|x) & D(q) < D(k) \end{cases}" /></a>

Evaluating p(k|x) and its gradient involve multiplying small (in range of [0,1]) numbers together multiple times, which may lead to underflow problems. Let L(k|x) = ln(p(k|x)) -- since ln is monotone decreasing, it should not change the location of optima when used in place of p(k|x) in the GINI expectation. L(k|x), is defined as:


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;L(k|x)&space;=&space;L(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;L(\textup{root}|x)&space;=&space;ln(f_{\textup{root}}(x))&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;L(k|x)&space;=&space;ln(f_k(x))&space;&plus;&space;L(k'|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;L(k|x)&space;=&space;L(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;L(\textup{root}|x)&space;=&space;ln(f_{\textup{root}}(x))&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;L(k|x)&space;=&space;ln(f_k(x))&space;&plus;&space;L(k'|x)" title="\newline \textup{Boundary Conditions:} \newline L(k|x) = L(k'|x) \textup{ if } l(k), \newline L(\textup{root}|x) = ln(f_{\textup{root}}(x)) \newline \newline \textup{Recursive Condition:} \newline L(k|x) = ln(f_k(x)) + L(k'|x)" /></a>


With gradient:


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(k|x)&space;=&space;\triangledown_&space;{\theta&space;_q}&space;L(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(\textup{root}|x)&space;=&space;1\{q=\textup{root}\}&space;\frac{1}{f_{\textup{root}}(x)}&space;\triangledown_&space;{\theta&space;_q}&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(k|x)&space;=&space;\begin&space;{cases}&space;0&space;&&space;D(q)>D(k)&space;\\&space;\frac&space;{1}{f_k(x)}&space;\triangledown&space;_{\theta&space;_q}&space;f_k(x)&space;&&space;D(q)&space;=&space;D(k)&space;\\&space;\triangledown&space;_{\theta&space;_q}&space;L(k'|x)&space;&&space;D(q)&space;<&space;D(k)&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\textup{Boundary&space;Conditions:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(k|x)&space;=&space;\triangledown_&space;{\theta&space;_q}&space;L(k'|x)&space;\textup{&space;if&space;}&space;l(k),&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(\textup{root}|x)&space;=&space;1\{q=\textup{root}\}&space;\frac{1}{f_{\textup{root}}(x)}&space;\triangledown_&space;{\theta&space;_q}&space;f_{\textup{root}}(x)&space;\newline&space;\newline&space;\textup{Recursive&space;Condition:}&space;\newline&space;\triangledown_&space;{\theta&space;_q}&space;L(k|x)&space;=&space;\begin&space;{cases}&space;0&space;&&space;D(q)>D(k)&space;\\&space;\frac&space;{1}{f_k(x)}&space;\triangledown&space;_{\theta&space;_q}&space;f_k(x)&space;&&space;D(q)&space;=&space;D(k)&space;\\&space;\triangledown&space;_{\theta&space;_q}&space;L(k'|x)&space;&&space;D(q)&space;<&space;D(k)&space;\end{cases}" title="\newline \textup{Boundary Conditions:} \newline \triangledown_ {\theta _q} L(k|x) = \triangledown_ {\theta _q} L(k'|x) \textup{ if } l(k), \newline \triangledown_ {\theta _q} L(\textup{root}|x) = 1\{q=\textup{root}\} \frac{1}{f_{\textup{root}}(x)} \triangledown_ {\theta _q} f_{\textup{root}}(x) \newline \newline \textup{Recursive Condition:} \newline \triangledown_ {\theta _q} L(k|x) = \begin {cases} 0 & D(q)>D(k) \\ \frac {1}{f_k(x)} \triangledown _{\theta _q} f_k(x) & D(q) = D(k) \\ \triangledown _{\theta _q} L(k'|x) & D(q) < D(k) \end{cases}" /></a>


