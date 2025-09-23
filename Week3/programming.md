## Problem
Find a model $\hat{f}$ to approximate $f(x)=\frac{1}{1+25x^{2}}$; $f^{'}(x)= -\frac{50x}{(1+25x^{2})^{2}}$, for $x \in [-1,1]$  

## Tranning model 
1 hidden layer ,tanh neural network.  
Why we use this model? Since runge function is smooth for $x \in [-1,1]$, by the [lemma](),then we can find a function $\hat{f}$ to approximate it.
Define $\hat{f}(x)= a_{0}+ \sum_{j=1}^{m}v_{j}tanh(w_{j}x+b_{j})$; $\frac{d \hat{f}}{dx}=\sum_{j=1}^{m}v_{j}(1-tanh^{2}(w_{j}x+b_{j}))w_{j}$， with $m=30$

## Data
1) Choose N trainning point by using unifrom nodes.
2) compute $y_{i} = f(x_{i})$ and $d_{i}=f'(x_{i})$.

## Loss
$\mathfrak{L}(\theta)=\lambda_{f} \cdot\frac{1}{N}\sum_{i}(\hat{f}(x_{i})-y_{i})^{2}+\lambda_{d}\cdot \frac{1}{N}\sum_{i}(\frac{d\hat{f}}{dx}(x_{i})-d_{i})^{2}$.  
<img width="500" height="350" alt="{FD34754E-DE87-400E-8762-A190CC208183}" src="https://github.com/user-attachments/assets/d8ef05e8-b84d-450b-8f71-0a5fe56b685b" />

## Result
<img width="450" height="350" alt="{43DF26EE-1F35-49A2-8238-C889E73DDADF}" src="https://github.com/user-attachments/assets/91a72b40-8c2b-4546-9930-dac956e4b204" />
<img width="450" height="350" alt="{C56CD8D1-D3FA-43E9-AD96-63ED96E77800}" src="https://github.com/user-attachments/assets/06b70218-b436-4880-b79c-4b81d574969a" />

