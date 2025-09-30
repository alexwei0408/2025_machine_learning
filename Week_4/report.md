## 訓練過程
1）讀入資料
使用function load把資料讀入進陣列。並且把位置初始化在$(120.00,21.88)$。
讀入資料後後，我們進行資料轉換
## i)定義類別
給定第i個節點的溫度，定義
$$
M_{i}=
\left\{
    \begin{matrix}
	    &1,\text{t}_{i} \text{ is valid} \\
        &0,\text{oterhwise }
    \end{matrix}
\right.
$$
## ii)Classification(for $h'_{0}$)
定義$y_{i}=M_{i}$,使其滿足資料格式$(lon_{i},lat_{i},y_{i})$,其中$y_{i} \in \left\{0,1\right\}$
## iii)Regression(for $h_{0}$)
保留有效節點$M_{i}=1$，回傳值為其原始溫度，定義為$value_{i}=t_{i}$ ，當$M_{i}=1$ 。其資料格式滿足$lon_{i},lat_{i},value_{i}$，並且將所有$M_{i}=0$的節點捨棄不用。

小結：通過步驟（i）,（ii），（iii）我們能得到兩個資料集。

---
2）標準化經緯度
$\mu_{j}=\frac{1}{N}\sum_{i=1}^{N}x_{ij}$   $\sigma_{j}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_{j})}$
For each data we take,
$$
z_{ij}=\frac{x_{ij}-\mu_{j}}{\sigma_{j}}
$$
Remark:從資料看出經緯度的數值範圍不同，若不進行標準化，將會造成梯度下降不穩定。

---
3）訓練模型
得到兩個資料集後，分別訓練一個簡單的機器學習模型
## i)classification model
模型： Logistic Regression
Input：經度，緯度
Output： $h'_{0}(x)=p=p(y=1|x)= \sigma(w^{T}x+b), 0\le p \le 1$.
Sigmoid function:$\sigma(x)=\frac{1}{1+e^{-x}}, \lim_{x\to -\infty} \sigma(x)=0,\lim_{x\to \infty} \sigma(x)=1$ and  $\sigma(0)=\frac{1}{2}$.
Loss Function: $L=-\frac{1}{N} \sum_{i=1}^{N}[y^{(i)}ln'_{0}(x^{(i)})+(1-y^{i})ln(1-h'_{0}(x^{(i)})]$.

## ii)Regression model
model: Linear Regression
Input: 經度，緯度
output：$h_{0}(x,y)=w_{0}​+w_{1} \cdot ​x+w_{2} \cdot y$。
data point: Take the point with $M_{i}=1$.
target: Find its minimum MSE 
$$
L=\frac{1}{N}\sum_{i=1}^{N}(h_{0}(x^{(i)},y^{(i)}-T^{(i)})^{2}+\lambda\left\| w \right\|^{2}_{2}
$$

---
4）合併模型
