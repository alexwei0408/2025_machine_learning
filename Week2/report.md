Problem description  
Use a neural network to approximate the Rugne function

$$
f(x)=\frac{1}{1+25x^{2}},x\in \left[-1,1\right]
$$  

i)The algorithms used for approximate runge function is Locally Weighted linear regression.  
  The weights is given by 

$$
w_i(x) = \exp(-\frac{(x^i-x)^2}{2\tau^2})
$$  

The loss is given by

$$
L\left( \theta;x_{q} \right)=\sum_{i=1}^{n}w_{i}\left( x_{q} \right)\left( y_{i}-\theta^{T}x_{i} \right)^{2}
$$

The prediction at $x_{q}$ is 

$$
\hat{y}\left( x_{q} \right)=\theta\left( x_{q} \right)^{\top}x_{q}
$$

ii)Node Selection  
I have choosed uniform nodes and chebyshev nodes of the first kind be my trainning points with the number of tranning points is 500. The following picture is the results of tranning.  
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/week2/image/results.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="600">
