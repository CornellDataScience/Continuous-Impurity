# Continuous-Impurity
Goal: To define a continuous analogue to an impurity metric so that decision trees can be trained with calculus-trained split-planes that are not forced to be parallel to an axis. A continuous analogue of GINI impurity, defined as:


<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" title="G = 1 - \frac{1}{|X|} \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} 1\left \{ f(X_i) = k\right \})^2} {\sum_{i} 1\left \{ f(X_i)=k\right \}}" /></a>

A continuous variant was determined by taking it's expectation -- considering the placement of a given input, x, into a given subset, k -- as being probabilistic in nature, dictated by the conditional probability p(k|x). p(k|x) is used when calculating the expectation of each indicator variable, and is the model to be trained when minimizing the expected impurity.

<a href="https://www.codecogs.com/eqnedit.php?latex=E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" title="E(G) = 1 - \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} p(k|X_i))^2} {|X|\sum_{i} p(k|X_i)}" /></a>


For use in caculus-optimized models, the gradient of the expected GINI impurity with respect the parameters of the model must be calculated, allowing for an arbitrary model to be trained with expected GINI by replacing the definition of p(k|x) and its gradient with respect to its parameters, theta.


<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;\textup{where}&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;\textup{where}&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" title="\newline \triangledown _\theta E(G) = \frac{-1}{|X|} \sum_{k}v_k(\triangledown _\theta u_k) + u_k (\triangledown _\theta v_k) \newline \textup{where} \newline u_k = \frac {1}{\sum_ i p(k|X_i)} \newline \triangledown _\theta u_k = \frac {-\sum _i \triangledown _\theta p(k|X_i)} {(\sum _i p(k|X_i))^2} \newline v_k = \sum _l (\sum _{i|y_i=l} p(k|X_i))^2 \newline \triangledown _\theta v_k = 2 \sum _l (\sum _{i|y_i = l} p(k|X_i)) \sum _{i|y_i = l} \triangledown _\theta p(k|X_i)" /></a>



So far, continuous impurity has been been applied to the following forms of p(k|x) and their associated required gradients:

Logistic Regression-Style p(k|x):

<a href="https://www.codecogs.com/eqnedit.php?latex=p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;x}}&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;x})^{-1}}{\partial&space;({\theta^Tx})}&space;\triangledown_\theta&space;(\theta^Tx)&space;=&space;p(k|x)(1-p(k|x))x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(k|x)&space;=&space;\frac&space;{1}{1&plus;e^{-\theta^T&space;x}}&space;\newline&space;\triangledown&space;_\theta&space;p(k|x)&space;=&space;\frac&space;{\partial&space;(1&plus;e^{-\theta^T&space;x})^{-1}}{\partial&space;({\theta^Tx})}&space;\triangledown_\theta&space;(\theta^Tx)&space;=&space;p(k|x)(1-p(k|x))x" title="p(k|x) = \frac {1}{1+e^{-\theta^T x}} \newline \triangledown _\theta p(k|x) = \frac {\partial (1+e^{-\theta^T x})^{-1}}{\partial ({\theta^Tx})} \triangledown_\theta (\theta^Tx) = p(k|x)(1-p(k|x))x" /></a>

Results: 


![Logistic 1](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Continuous%20impurity%20first%20working%20result.png?raw=true)





A beneficial trait of models trained with expected GINI is that they can be used as the "split" model of decision trees, allowing more expressive decision boundaries than what was previously possible. Some examples of using the Logistic Regression-style p(k|x) as the split function for a decision tree:

![Continuous Decision Tree 1](https://github.com/CornellDataScience/Continuous-Impurity/blob/master/Continuous%20Imuprity%20Decision%20Tree.png?raw=true)

![Continuous Decision Tree 2](https://raw.githubusercontent.com/CornellDataScience/Continuous-Impurity/master/Continuous%20Imuprity%20Decision%20Tree.png)


