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
