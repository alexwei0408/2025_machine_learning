## 主要目標: 學習 PDF $p(x)$

想法：對於 Score-Based generative model來說就是學習pdf，但是我們很難建構出一個擁有良好性質的pdf（i. $p(x)>0$ ;ii. $\int_{\mathbb{R}}p(x)dx=1$ ）
所以我們改來學習Score function。假設 $p(x;\theta)=\frac{e^{q(x;\theta)}}{z(\theta)}$，得出 $\log(p;\theta)=q(x;\theta)-\log z(\theta)$，

$$
S(x)=\nabla_{x} \log(p(x))=\nabla_{x} \log q(x;\theta)。
$$


以下為的S(x)的Loss function，也被稱作Explicit Score Matching，

$$
L_{ISM}(\theta)=\mathbb{E}_{x\sim p(x)}||S(x;\theta)-\nabla_{x}\log p(x)||^{2}。
$$

但是對於 $\nabla_{x} \log p(x)$，我們仍是不知道，所以通過以下有幾個方法能學習 $\nabla_{x} \log p(x)$。  


i）我們通過一些計算把Loss Function等價的寫成

$$
L_{ISM}(\theta)=\mathbb{E}_{x\sim p(x)}(||S(x;\theta)||^{2}+2\nabla_{x}\cdot S(x;\theta))。
$$

我們把其稱為 Implicit Score Matching。

Remark  
a）使用ISM，我們不需要知道normalize constant Z，因為 $\nabla_{x} \log Z=0$。  
b）直接作用於原始資料 $p(x)$。  
c）ISM在高緯度計算瓶頸（eg如果資料模型是 $10^{5}x10^{5},我們就需要計算這麼多次$），因而有人提出方法（ii）DSM。    

---

ii）我們先定義以下符號

$$
\begin{aligned}
&x_{0}:\text{orignal data;}& \\
&p_{0}(x_{0}): \text{data distribution of original data;} \\
&x:\text{noisy data (by perturbing the orignal data);} \\
&p(x|x_{0}):\text{conditional (noisy) data distribution;} \\
&p_{\sigma}(x):\text{(noisy) data distribution.}
\end{aligned}
$$

定義 $p_{\sigma}(x)= \int_{\mathbb{R^{d}}}p(x|x_{0})p_{0}(x_{0})dx_{0}$，我們把其稱作noisy score function。

$$
L_{DSM}(\theta) = \mathbb{E}_{x_0\sim p_0(x_0)}\mathbb{E}_{x|x_0\sim p(x|x_0)}\left[\|S_\sigma(x;\theta)-\nabla_{x}\log p(x|x_0)\|^2\right],
$$

如上loss function被稱作 Denoising Score Matching。

最終我們要學習 $\nabla_{x} \log p(x)$，但是DSM是透過雜訊化後的分布空間學分數，進而間接逼近乾淨分布的資訊。  

Remark  
a）設計已知 $p(x|x_{0})$ 使得目標 $\nabla_{x} \log p(x|x_{0})$ 能被訓練。  
b） $\nabla_{x} \log p_{\sigma}(x)$ 與 $\nabla_{x} \log p(x|x_{0})$ ，兩者關係可以近似。 
c）模型學習的是從雜訊還原的方向，對於實作更為容易。  

---

我們透過上述方法學習了 $S(x)$ 後， 便可以用於生成新的樣本

流程  
1）從簡單的噪聲分佈開始(eg $x_{T}\sim \mathcal{N}(0,I)$ )。   
2）沿著逆向時間進行迭代。每次使用模型 $S(x_{t},t)$ 所預測的 Score 作為“往高密度方向移動”的指引，並加入適量的隨機噪聲，保持模型生成樣本的探索性。  
3）當雜訊強度逐漸降至零時，最終逼近資料分布。  


```
pseudo-code

# 訓練完成後，使用 s_theta(x, t) 生成樣本

# 1. 從最強雜訊開始
x = sample_noise(shape = data_shape)  # e.g., x ~ Normal(0, I)

# 2. 逆時間循環 t 从 T → 0
for t = T, T−Δt, T−2Δt, …, Δt:
    sigma_t = compute_noise_scale(t)
    
    # 使用模型預測分數
    score_pred = s_theta(x, t)
    
    # 更新樣本（例如用 Euler–Maruyama 法則，簡化版）
    x = x + step_size * score_pred \
        + sqrt(2 * step_size) * noise()
    
    # （可選）加上雜訊縮小／正則化項
    
# 3. 最後獲得 x_0 ≈ 來自資料分布的樣本
return x
```

Remark  
a）前向加噪聲假設將資料分布漸變為已知噪聲分布；而其對應的逆向 SDE 僅依賴時間條件下的分數函數 $\nabla_{x} \log p_{t}(x)$，透過估計該分數並解逆向方程，即可實現從噪聲到資料的生成。
