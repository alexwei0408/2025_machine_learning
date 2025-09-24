## (Lemma 3.1)   
Let $k \in \mathbb{N}$ and $s \in 2\mathbb{N}-1$. Then it holds that for all $\epsilon>0$ there exists a shallow tanh neural network 
$\Phi_{s,\epsilon}:[-M,M] \to \mathbb{R}^{\frac{s+1}{2}}$ of width $\frac{s+1}{2}$ such that

$$
\max_{\substack{p\le s,p odd}}\left\| f_{p}-(\Phi_{s,\epsilon})_{\frac{p+1}{2}} \right\|_{W^{k,\infty}}\le \epsilon.
$$  

Moreover, the weights of $\Phi_{s,\epsilon}$ scale as $O(\epsilon^{\frac{-s}{2}}(2(s+2)) \sqrt{2M})^{s(s+3)}$ for small $\epsilon$ and large $s$.

## Discussion  
Can tanh neural networks approximate any smooth function? If it can approximate, what is the strategy to approximate it.  
1. Approximate polynomial.
2. Since for all smooth function, it can be product by ${1,x,x^{2},...}$. Then polynomial can use for approximate smooth function.

For $x \to 0$, we have $tanh(x) \simeq x$.
## Statement  
For any integer s, we can build a shallow tanh network of width $\frac{s+1}{2}$ that approximate odd functin $x,x^{3},...$

## Example  
If $s=3$, then we can approximate both $x$ and $x^{3}$

---
## (Lemma 3.2)
Let $k \in \mathbb{N}, s \in 2\mathbb{N}-1$ and  $M>0$. For every $\epsilon >0$, there exists a shallow tanh neural network $\Psi_{s,\epsilon}:[-M,M] \to \mathbb{R}^{s}$ of width $\frac{3(s+1)}{2}$ such that

$$
\max_{\substack{p \le s}} \left\| f_{p}-(\Psi_{s,\epsilon})_{W^{k,\infty}} \right\|\le \epsilon
$$

## Statement
When we use the binomial combination of shifted tanh function, we will get a function that is very close to $t^{p}$ evevrywhere on the interval, and its derivatives up to oerder k are also close.


---
Conclusion: Lemma3.1-3.2 show that tanh networks can build monomials $x^{p}$ accurately.
---
