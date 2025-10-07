1. Given

$$
f(x)=\frac{1}{\sqrt{(2 \pi)^{k} |\Sigma| }}e^{{-\frac{1}{2}}(x-\mu)^{T}\Sigma^{-1}(x-\mu)},
$$

where $x,\mu \in \mathbb{R}^{k}, \Sigma$ is a $k-by-k$ positive definite matrix and $|\Sigma|$ is its determinant. Show that $\int_{\mathbb{R}^{k}}f(x)dx=1$.

---
sol  
Since $\Sigma$ is a k-by-k positive definite matrix. Then, there exists an invertible lower-triangular matrix L such that $\Sigma=LL^{T}.$ Therefore,

$$
	|\Sigma|=|L|^{2},\Sigma^{-1}=(L^{T})^{-1}L^{-1}.
$$

Define $y=L^{-1}(x-\mu)$, and so we have $x=\mu+Ly$.
Thus,

$$
\begin{align}
\int_{\mathbb{R}^{k}}\frac{1}{\sqrt{(2 \pi)^{k} |\Sigma| }}e^{{-\frac{1}{2}}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}dx 
&=\frac{1}{\sqrt{(2 \pi)^{k} |\Sigma| }}\int_{\mathbb{R}^{k}}e^{-\frac{1}{2}y^{T}y}|L|dy \\
&=\frac{1}{(2\pi)^{\frac{k}{2}}}\int_{\mathbb{R}^{k}}e^{-\frac{1}{2}\sum_{i=1}^{k}y_{i}^{2}}dy \\
&=\prod_{i=1}^{k}(\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{-t^{2}}dt) \\
&=\prod_{i=1}^{k}1=1.
\end{align}
$$

---

2.Let A,B be n-by-n matrices and x be a n-by-1 vector.  
	a) Show that $\frac{\partial }{\partial A}trace(AB)=B^{T}.$  
	b) Show that $x^{T}Ax=trace(xx^{T}A).$  
	c) Derive the maximum likelihood estimators for a multivariate Gaussian.   
	
  sol
	a)Take $f(A)=tr(AB)$. Then,
  
  $$
	\begin{align}
	\frac{\partial }{\partial A}trace(AB)=trace(\frac{\partial }{\partial A}AB) 
	&= trace(\partial_{A}AB+A\partial_{A}B) \\
	&= trace(\partial_{A}AB).
	\end{align}
	$$
  
  Since $trace(AB)=trace(BA).$ Then $trace(\partial_{A}AB)=trace(B\partial_{A}A)$. By inner product of a vector and a gradient,  we have $\partial_{A}f=<\nabla_{A}f,dA>_{F}=tr((\nabla_{A}f)^{T}dA$, then  $\nabla_{A}f=B^{T}.$  
	b)By cyclic property of trace, we have
  
  $$
	x^{T}Ax=tr(x^{T}Ax)=tr(Axx^{T})=tr(Axx^{T}).
	$$
	
  c) By Log-likelihood function, we have $l(\mu,\Sigma)=-\frac{N}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^{N}(x_{i}-\mu)^{T}\Sigma^{-1}(x_{i}-\mu).$
	From (b), we get $l(\mu,\Sigma)=-\frac{N}{2}\log|\Sigma|-\frac{1}{2}\sum_{i=1}^{N}tr(\Sigma^{-1}(x_{i}-\mu)(x_{i}-\mu)^{T}).$ First, we derivative with vector $\mu$, then
	
   $$
	 \begin{align}
	 \nabla_{\mu}l &=\nabla_{\mu}\frac{1}{2}\sum_{i=1}^{N}tr(\Sigma^{-1}(x_{i}-\mu)(x_{i}-\mu)^{T})\\
	 &=\nabla_{\mu}\frac{1}{2}\sum_{i=1}^{N}(tr(\Sigma^{-1}x_{i}x_{i}^{T})-tr(\Sigma^{-1}x_{i}\mu^{T})-tr(\Sigma^{-1}\mu x^{T}_{i})+tr(\Sigma^{-1}\mu \mu^{T})) \\
	 &=\nabla_{\mu}\frac{1}{2}\sum_{i=1}^{N}(-tr(\Sigma^{-1}x_{i}\mu^{T})-tr(\Sigma^{-1}\mu x_{i})+tr(\Sigma^{-1}\mu \mu^{T}))
	 \end{align}
	 $$
	
  From (a)  
  
  $$
	 \begin{align}
	 \nabla_{\mu}(tr(\Sigma^{-1}x_{i}\mu^{T})) &=-\Sigma^{-1}x; \\ 
	 \nabla_{\mu}tr(\Sigma^{-1}\mu x^{T}_{i})&= -\Sigma^{-1}x; \\
	 \nabla_{\mu}tr(\Sigma^{-1}\mu \mu^{T}) &= \nabla_{\mu}tr(\mu^{T}\Sigma^{-1}\mu)=2\Sigma^{-1}\mu.
	 \end{align}
  $$
   
   Therefore $\nabla_{\mu}l=\Sigma^{-1}\sum_{i=1}^{N}(x_{i}-\mu).$ We set $\sum_{i=1}^{N}(x_{i}-\mu)=0$, which implies that $\hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}x_{i}$.  
	 Second, we derivative with $\Sigma$. By  
   <img width="548" height="51" alt="{CAE0A6A9-05D0-4517-B341-0646CA847EFE}" src="https://github.com/user-attachments/assets/9076b96f-1c44-4bee-9127-5d41ddd884fb" />  
   and, 
   
   $$
   \begin{align}
   &　d(AA^{-1})=dI=0 \\
   &　dA(A^{-1})+Ad(A^{-1})=0 \Rightarrow　d(A^{-1})= -A^{-1}(dA)A^{-1}.
   \end{align}
   $$

   then we get $dlog|\Sigma|=tr(\Sigma^{-1}d\Sigma)$, and $d\Sigma^{-1}=-\Sigma^{-1}d\Sigma \Sigma^{-1}$, and so 
	 
   $$
	 \begin{align}
	 \nabla_{\Sigma}l &= -\frac{N}{2}tr(\Sigma^{-1}d\Sigma)+\frac{1}
	 {2}tr(\Sigma^{-1}\sum_{i=1}^{N}(x_{i}-\mu)(x_{i}-\mu)^{T}\Sigma^{-1}d\Sigma) \\
	 \text{then } 0&= -\frac{N}{2}\Sigma^{-1}+\frac{1}{2}\Sigma^{-1}\sum_{i=1}^{N}(x_{i}-\mu)(x_{i}-\mu)^{T}\Sigma^{-1} \Rightarrow  N\Sigma=\sum_{i=1}^{N}(x_{i}-\mu)(x_{i}-\mu)^{T}
	 \end{align}
	 $$
   
Therefore, $\hat{\Sigma}=\frac{1}{N}\sum_{i=1}^{N}(x_{i}-\hat{\mu})(x_{i}-\hat{\mu})^{T}$. 
