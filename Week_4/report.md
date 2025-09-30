## 訓練過程

### 1）讀入資料
使用 function `load` 把資料讀入進陣列，並且把位置初始化在 $(120.00, 21.88)$。  
讀入資料後，我們進行資料轉換。

---

### i) 定義類別
給定第 $i$ 個節點的溫度，定義：

$$
M_{i} =
\begin{cases}
1, & t_{i} \text{ is valid} \\
0, & \text{otherwise}
\end{cases}
$$

---

### ii) Classification (for $h'_{0}$)
定義 $y_{i} = M_{i}$，使其滿足資料格式 $(lon_{i}, lat_{i}, y_{i})$，其中 $y_{i} \in \{0, 1\}$。

---

### iii) Regression (for $h_{0}$)
保留有效節點 $M_{i}=1$，回傳值為其原始溫度，定義為 $value_{i} = t_{i}$，當 $M_{i}=1$。  
其資料格式滿足 $(lon_{i}, lat_{i}, value_{i})$，並且將所有 $M_{i}=0$ 的節點捨棄不用。

---

**小結**：通過步驟（i）、（ii）、（iii），我們能得到兩個資料集。

---

### 2）標準化經緯度
均值與標準差：

$$
\mu_{j} = \frac{1}{N}\sum_{i=1}^{N} x_{ij}, 
\qquad
\sigma_{j} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_{j})^{2}}
$$

對每筆資料取：

$$
z_{ij} = \frac{x_{ij}-\mu_{j}}{\sigma_{j}}
$$

**Remark**: 從資料看出經緯度的數值範圍不同，若不進行標準化，將會造成梯度下降不穩定。

---

### 3）訓練模型
得到兩個資料集後，分別訓練一個簡單的機器學習模型。

#### i) Classification model
- 模型：Logistic Regression  
- Input：經度，緯度  
- Output： $h'_{0}(x) = p = p(y=1|x) = \sigma(w^{T}x+b), \quad 0 \leq p \leq 1$

- Sigmoid function：

$$
\sigma(x) = \frac{1}{1+e^{-x}}, 
\quad \lim_{x\to -\infty} \sigma(x)=0, 
\quad \lim_{x\to \infty} \sigma(x)=1, 
\quad \sigma(0)=\tfrac{1}{2}
$$

- Loss Function：

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y^{(i)} \ln h'_{0}(x^{(i)}) + (1-y^{(i)}) \ln \big(1-h'_{0}(x^{(i)})\big) \Big]
$$

---

#### ii) Regression model
- 模型：Linear Regression  
- Input：經度，緯度  
- Output： $h_{0}(x,y) = w_{0} + w_{1}x + w_{2}y$
  
- Data point：取 $M_{i}=1$ 的點。  
- 目標：最小化 MSE。  

$$
L = \frac{1}{N} \sum_{i=1}^{N}\Big(h_{0}(x^{(i)},y^{(i)}) - T^{(i)}\Big)^{2} + \lambda \| w \|^{2}_{2}
$$

---

### 4）合併模型
