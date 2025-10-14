1a) [code](https://github.com/alexwei0408/2025_machine_learning/blob/main/Week_6/GDA.py)  
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

```
python
import numpy as np

class GDA:
    def __init__(self, shared_cov=True, reg=1e-6):
        self.shared_cov, self.reg = shared_cov, reg

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, int)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.pi_ = []; self.mu_ = []; self.S_ = []
        for k in self.classes_:
            Xk = X[y==k]; self.pi_.append(len(Xk)/n)
            mu = Xk.mean(axis=0); self.mu_.append(mu)
            Sk = (Xk - mu).T @ (Xk - mu) / len(Xk) + self.reg*np.eye(d)
            self.S_.append(Sk)
        self.pi_, self.mu_ = np.array(self.pi_), np.vstack(self.mu_)
        if self.shared_cov:  # pooled covariance for LDA
            nks = np.array([(y==k).sum() for k in self.classes_])
            pooled = sum(nk*(Sk - self.reg*np.eye(d)) for nk,Sk in zip(nks,self.S_))
            self.Sshared_ = pooled / n + self.reg*np.eye(d)
        return self

    def _g(self, X):
        X = np.asarray(X, float)
        out = np.empty((len(X), len(self.classes_)))
        for i, k in enumerate(self.classes_):
            mu = self.mu_[i]
            S  = self.Sshared_ if self.shared_cov else self.S_[i]
            xc = X - mu
            # log N(x; mu, S) up to constant
            L  = np.linalg.cholesky(S)
            z  = np.linalg.solve(L, xc.T)
            quad = (z*z).sum(axis=0)
            logdet = 2*np.log(np.diag(L)).sum()
            out[:, i] = -0.5*(logdet + quad) + np.log(self.pi_[i])
        return out

    def predict(self, X):
        return self._g(X).argmax(axis=1)
```

c)Accuracy on this data set.  
<img width="900" height="600" alt="learning_curve" src="https://github.com/user-attachments/assets/000711a6-55a3-431d-a15a-ad54dfc2366a" />

d)Decision boundary  
<img width="900" height="600" alt="boundary_qda" src="https://github.com/user-attachments/assets/082c1eac-3a94-49d3-a629-9f3e014ef09d" />
```
python
# Decision boundary plot (for QDA or LDA)
import matplotlib.pyplot as plt
def plot_boundary(model, X, y, xlim, ylim, n=400):
    xs = np.linspace(*xlim, n); ys = np.linspace(*ylim, n)
    gx, gy = np.meshgrid(xs, ys)
    Z = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.figure()
    plt.contourf(gx, gy, Z, levels=[-0.5,0.5,1.5], alpha=0.25)
    plt.contour(gx, gy, Z, levels=[0.5], linewidths=1.2)
    plt.scatter(X[y==1,0], X[y==1,1], s=6, label="valid (1)", alpha=0.7)
    plt.scatter(X[y==0,0], X[y==0,1], s=6, label="invalid (0)", alpha=0.7)
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend(); plt.title("GDA boundary")

```

Remark: Why GDA fits this data set?  
Since the features are onlly two-dimensional and sizes of data set is large, then we can estimate $\mu_{k}$ and $\Sigma_{k}$ in very stable way.
Beside that, the positive class (valid/land) forms a single, roughly elliptical cluster in the plane, while the negative class lies around it.
This geometry can be well approximated and enclosed by QDA’s quadratic decision boundary.

Final result  
<img width="900" height="600" alt="validity_map" src="https://github.com/user-attachments/assets/592bb403-838e-4f7a-b08f-7c0073893626" />

2)Problem setting Given $C(x) \in \{ 0,1 \}$ be classification model, and $R(x) \in \mathbb{R}$ be regression model and define a combine function

$$
h(x)=
\begin{cases}
R(x) &,C(x)=1 \\
-999 &,C(x)=0 \\
\end{cases}
$$

Regression model: Bilinear interpolating

[Result](https://github.com/user-attachments/files/22913568/predictions.xlsx)
