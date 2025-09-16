Problem description  
Use a neural network to approximate the Rugne function

$$
f(x)=\frac{1}{1+25x^{2}},x\in \left[-1,1\right]
$$  

<u> What algorithms used for approximate </u>  
That's Locally Weighted linear regression.The weights is given by 

$$
w_i(x) = \exp(-\frac{(x^i-x)^2}{2\tau^2})
$$
