
1)
b)Problem setting.
  We want to classify each grid location by its longtitde_latitude feature $x \in \mathbb{R}^ {2}$ into

$$
y=
\begin{cases}
1 ,&\text{valid temperature}\\
0 ,&\text{invalid temperature}\\
\end{cases}
$$

We want to use GDA to be my classification model. GDA assumes a class-conditional Gaussian for each label:

$$
p(x|y=k)= \mathcal{N} (x; \mu_{k}; \Sigma_{k}), \pi_{k}=P(y=k)
$$  

Then, we use Bayes' Rule, the classifier predicts the class with the largest posterior $\hat{y}=\arg\max_{k}(\log \pi_{k}+ \log \mathcal{N}(x;\mu_{k};\Sigma_{k}))$, 
wehre $\pi_{k}$ is the class frequency; $\mu_{k}$ is the sample mean of class, k; $\Sigma$ is the sample convaraiance.  
priors: $\hat{\pi_{k}}=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}(y_{i}=k)$;  
means: $\mu_{k}=\frac{1}{n_{k}}\sum_{i:y_{i}=k}x_{i}$;  
convariances: $\hat{\Sigma}=\frac{1}{n_k}\sum_{i:\, y_i = k}\bigl(x_i-\hat{\mu}_k\bigr)\bigl(x_i-\hat{\mu}_k\bigr)^{T}$

c)Accuracy on this data set.  
<img width="900" height="600" alt="learning_curve" src="https://github.com/user-attachments/assets/000711a6-55a3-431d-a15a-ad54dfc2366a" />

d)Decision boundary  
<img width="900" height="600" alt="boundary_qda" src="https://github.com/user-attachments/assets/082c1eac-3a94-49d3-a629-9f3e014ef09d" />

Remark: Why GDA is suitable for this data set?  
Since the features are onlly two-dimensional and sizes of data set is large, then we can estimate $\mu_{k}$ and $\Sigma_{k}$ in very stable way.
Beside that, the positive class (valid/land) forms a single, roughly elliptical cluster in the plane, while the negative class lies around it.
This geometry can be well approximated and enclosed by QDA’s quadratic decision boundary.
