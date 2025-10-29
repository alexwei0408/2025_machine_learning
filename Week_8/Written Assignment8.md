## 1) Show that the sliced score matching (SSM) can also be written as 

$$
L_{SSM}= \mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)}\left\[\left\|v^{T}S(x;\theta)\right\|^{2}+2v^{T}\nabla_{x}(v^{T}S(x;\theta)) \right\]. -(\star)
$$

sol:

Note that we have 

$$
L_{SSM}=\mathbb{E}_{x\sim p(x)}\mathbb{E}_{v\sim p(v)} \left\[ \left\|v^{T}S(x;\theta)\right\|^{2} + 2v^{T}\nabla_{x}(v^{T}S(x;\theta))\right\].
$$ 

By linearity of expectation, then we can rewrite $(\star)$ into 

$$
\begin{aligned}
\mathbb{E}_{x\sim p(x)}(\mathbb{E}_{v\sim p(v)}(v^{T}S)^{2})&=\mathbb{E}_{x\sim p(x)}(\mathbb{E}_{v\sim p(v)}(v^{T}(SS^{T})v)) \\
&=\mathbb{E}_{x\sim p(x)}(\mathbb{E}_{v\sim p(v)}(Tr(SS^{T})(vv^{T})) \\
&= \mathbb{E}_{x\sim p(x)}Tr(SS^{T}(\mathbb{E}_{v\sim p(v)}[vv^{T}])
\end{aligned}
$$

Since $\mathbb{E}_{v}\left\[vv^{T}\right\] = I$ for $v \sim \mathcal{N}(0, I)$, then we get 

$$
\mathbb{E}_{x\sim p(x)}Tr(SS^{T}(\mathbb{E}_{v\sim p(v)}[vv^{T}])=Tr(SS^{T})=\mathbb{E}_{x\sim p(x)}\left\|S\right\|^{2},
$$

and so we completed the proof.

---

## 2) Briefly explain SDE.



