# Continuous-Impurity
For defining and testing a continuous impurity cost

GINI Impurity:


<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" title="G = 1 - \frac{1}{|X|} \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} 1\left \{ f(X_i) = k\right \})^2} {\sum_{i} 1\left \{ f(X_i)=k\right \}}" /></a>


Expected GINI Impurity:

<a href="https://www.codecogs.com/eqnedit.php?latex=E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" title="E(G) = 1 - \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} p(k|X_i))^2} {|X|\sum_{i} p(k|X_i)}" /></a>
