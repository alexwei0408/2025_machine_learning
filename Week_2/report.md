Problem description  
Use a neural network to approximate the Rugne function

$$
f(x)=\frac{1}{1+25x^{2}},x\in \left[-1,1\right]
$$  

i)Algorithms  
The algorithms used for approximate runge function is Locally Weighted linear regression.  
  The weights is given by 

$$
w_i(x) = \exp(-\frac{(x_{i}-x)^2}{2\tau^2})
$$  

where $\tau\gt 0$ is the bandwidth parameter controlling how local the fit is.In this report, we fixed $\tau =0.2$.  
The loss is given by

$$
L\left( \theta;x_{q} \right)=\sum_{i=1}^{n}w_{i}\left( x_{q} \right)\left( y_{i}-\theta^{T}x_{i} \right)^{2}
$$

The prediction at $x_{q}$ is 

$$
\hat{y}\left( x_{q} \right)=\theta\left( x_{q} \right)^{\top}x_{q}
$$

ii)Node Selection  
I have choosed uniform nodes and chebyshev nodes of the first kind be my trainning points with the number of tranning points is 150.   
The following picture is the results of tranning.  
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/Week_2/image/results.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="450">
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/Week_2/image/error.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="450">  
<u>a)Obsevertion</u>  
Both uniform nodes and chebyshev of the error near the boundaries is small, but at x=0, the approximation is not good.  
<u>b)How to improve</u>  
First we try to change the number of tranning points to 300. Then let $\tau$ become smaller.  
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/Week_2/image/2.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="300">
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/Week_2/image/3.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="300">
<img src="https://raw.githubusercontent.com/alexwei0408/2025_machine_learning/refs/heads/main/Week_2/image/1.png" alt="LWLR with Uniform vs Chebyshev Nodes" width="300">  
<u>Conclusion</u>  
The parameter $\tau$ controls the degree of smoothness of the LWLR. When $\tau$ is small, the model can accurately track the rapid changes of the Runge function at $x=0$.  
iii)Error  
a)Means Square Error: By using $MSE=\frac{1}{N}\sum_{i=1}^{N}\left( \hat{f}(x_{i})-f(x_{i}) \right)^{2}$,  
then we get $MSE=1.250* 10^{-2}$ for Uniform Nodes, and $MSE=1.197* 10^{-2}$ for cheyshev of the first kind.
