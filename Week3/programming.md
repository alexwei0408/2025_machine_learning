## Problem
Find a model $\hat{f}$ to approximate $f(x)=\frac{1}{1+25x^{2}}$; $f^{'}(x)= -\frac{50x}{(1+25x^{2})^{2}}$, for $x \in [-1,1]$  

## Tranning model 
1 hidden layer ,tanh neural network.  
Why we use this model?  
Since runge function is smooth for $x \in [-1,1]$, by the [lemma](https://github.com/alexwei0408/2025_machine_learning/blob/main/Week3/report.md),then we can find a function $\hat{f}$ to approximate it.  
Define $\hat{f}(x)= a_{0}+ \sum_{j=1}^{m}v_{j}tanh(w_{j}x+b_{j})$; $\frac{d \hat{f}}{dx}=\sum_{j=1}^{m}v_{j}(1-tanh^{2}(w_{j}x+b_{j}))w_{j}$， with $m=30$

## Data
1) Choose N trainning point $x_{i}$ by using unifrom nodes.
2) compute $y_{i} = f(x_{i})$ and $d_{i}=f'(x_{i})$.

## Loss
$\mathfrak{L}(\theta)=\lambda_{f} \cdot\frac{1}{N}\sum_{i}(\hat{f}(x_{i})-y_{i})^{2}+\lambda_{d}\cdot \frac{1}{N}\sum_{i}(\frac{d\hat{f}}{dx}(x_{i})-d_{i})^{2}$, with $\lambda_{f}=\lambda_{d}=1$.   
<img width="500" height="350" alt="{FD34754E-DE87-400E-8762-A190CC208183}" src="https://github.com/user-attachments/assets/d8ef05e8-b84d-450b-8f71-0a5fe56b685b" />  
<img width="300" height="220" alt="{C255BA0D-E401-4ACA-B1B2-852300A2B65F}" src="https://github.com/user-attachments/assets/8a2bad38-5b63-4d8e-87a0-cd89abbcc085" />
<img width="300" height="220" alt="{D42D6688-F7B1-407A-9422-B6292EC2A3FE}" src="https://github.com/user-attachments/assets/a4b9e80a-332a-41f6-bbd9-1fab246b3b1f" />

## Result
<img width="450" height="350" alt="{43DF26EE-1F35-49A2-8238-C889E73DDADF}" src="https://github.com/user-attachments/assets/91a72b40-8c2b-4546-9930-dac956e4b204" />
<img width="450" height="350" alt="{C56CD8D1-D3FA-43E9-AD96-63ED96E77800}" src="https://github.com/user-attachments/assets/06b70218-b436-4880-b79c-4b81d574969a" />

## MSE
$MSE_{f}=\frac{1}{N}\sum_{i=1}{N}(\hat{f}(x_{i})-f(x_{i}))^{2}$; $MSE_{f'}=\frac{1}{N}\sum_{i=1}{N}(\hat{f'}(x_{i})-f'(x_{i}))^{2}$  
<img width="923" height="71" alt="{4DB5BB82-1A28-41C0-A759-FB10DB6FA024}" src="https://github.com/user-attachments/assets/f96df6dc-a3e9-4ac2-a20b-e8609cdce29e" />  

## Conclusion
The MSE of runge function is $<10^{-5}$, and MSE of its deravitive is quick bigger that is $<10^{-3}$. For the region $x\simeq 0 \hat{f}$ is hard to match the true function, and the accurary of approximation to f' is worst.  
<img width="450" height="300" alt="{D03576B5-FA14-4A7F-A016-89F60D3CBB34}" src="https://github.com/user-attachments/assets/0a87e7f9-afd7-472e-99d3-26396d90e609" />
<img width="450" height="300" alt="{07D0E8E6-22D1-49D7-8B9F-4C71AB1AD180}" src="https://github.com/user-attachments/assets/2add9279-baed-4934-83fd-cf0e181dc233" />
