# Continuous-Impurity
For defining and testing a continuous impurity cost

GINI Impurity:


<a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G&space;=&space;1&space;-&space;\frac{1}{|X|}&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;1\left&space;\{&space;f(X_i)&space;=&space;k\right&space;\})^2}&space;{\sum_{i}&space;1\left&space;\{&space;f(X_i)=k\right&space;\}}" title="G = 1 - \frac{1}{|X|} \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} 1\left \{ f(X_i) = k\right \})^2} {\sum_{i} 1\left \{ f(X_i)=k\right \}}" /></a>


Expected GINI Impurity:

<a href="https://www.codecogs.com/eqnedit.php?latex=E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(G)&space;=&space;1&space;-&space;\sum_{k}&space;\frac&space;{\sum_&space;{l}&space;(\sum_{i|y_i&space;=&space;l}&space;p(k|X_i))^2}&space;{|X|\sum_{i}&space;p(k|X_i)}" title="E(G) = 1 - \sum_{k} \frac {\sum_ {l} (\sum_{i|y_i = l} p(k|X_i))^2} {|X|\sum_{i} p(k|X_i)}" /></a>


Gradient of Expected GINI Impurity: 





<a href="https://www.codecogs.com/eqnedit.php?latex=\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\newline&space;\triangledown&space;_\theta&space;E(G)&space;=&space;\frac{-1}{|X|}&space;\sum_{k}v_k(\triangledown&space;_\theta&space;u_k)&space;&plus;&space;u_k&space;(\triangledown&space;_\theta&space;v_k)&space;\newline&space;u_k&space;=&space;\frac&space;{1}{\sum_&space;i&space;p(k|X_i)}&space;\newline&space;\triangledown&space;_\theta&space;u_k&space;=&space;\frac&space;{-\sum&space;_i&space;\triangledown&space;_\theta&space;p(k|X_i)}&space;{(\sum&space;_i&space;p(k|X_i))^2}&space;\newline&space;v_k&space;=&space;\sum&space;_l&space;(\sum&space;_{i|y_i=l}&space;p(k|X_i))^2&space;\newline&space;\triangledown&space;_\theta&space;v_k&space;=&space;2&space;\sum&space;_l&space;(\sum&space;_{i|y_i&space;=&space;l}&space;p(k|X_i))&space;\sum&space;_{i|y_i&space;=&space;l}&space;\triangledown&space;_\theta&space;p(k|X_i)" title="\newline \triangledown _\theta E(G) = \frac{-1}{|X|} \sum_{k}v_k(\triangledown _\theta u_k) + u_k (\triangledown _\theta v_k) \newline u_k = \frac {1}{\sum_ i p(k|X_i)} \newline \triangledown _\theta u_k = \frac {-\sum _i \triangledown _\theta p(k|X_i)} {(\sum _i p(k|X_i))^2} \newline v_k = \sum _l (\sum _{i|y_i=l} p(k|X_i))^2 \newline \triangledown _\theta v_k = 2 \sum _l (\sum _{i|y_i = l} p(k|X_i)) \sum _{i|y_i = l} \triangledown _\theta p(k|X_i)" /></a>
