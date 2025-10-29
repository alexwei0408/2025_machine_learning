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

SDE的一般形式為

$$
dx_t = \underbrace{f(x_t, t)}_{\text{drift}}\,dt + \underbrace{G(x_t, t)}_{\text{diffusion}}\,dW_t, \quad x(0)=x_0,
$$

通過一般式能看出SDE是以布x朗運動為雜訊驅動的動態模型，其解具有隨機過程性質。我們能拆成兩項來理解這個這個一般式，分別是  
i） $f(x_{t},t)dt$ 系統中可預測的部分，代表 $x_{t}$s隨時間變化的期望路徑，被稱做Drift Term。  
ii） $G(x_{t},t)dW_{t}$ 系統中隨機部分，代表了系統的noise，被稱作Diffusion Term。若這項為零時，則系統簡化成ODE形式

特點： 給定一個IVP，通常ODE在解完後會給出一個唯一解，但是SDE的解是隨機過程 $x_{t}$,每步的增量 $\Delta W_{s}\sim \mathcal{N}(0,\Delta t)$不同，因此會產生不同路徑，故最後生成的解會像以下形式，  
<img width="782" height="513" alt="{9B9976A4-61DC-4515-A6A2-AEEBF7A1EBFA}" src="https://github.com/user-attachments/assets/11a489fa-9bb1-4683-bc8b-0a323f82c84b" />  

如何應用在diffusion model：SDE在生成模型就是一個反向的過程，從很多noise的圖片逆向走出一條路徑回到起點。

