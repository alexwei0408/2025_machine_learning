# combined_model_nolib.py
# 結合你的 classification (logistic) 與 regression (linear) → 新模型 (hard/soft gating)
# 只用標準函式庫。把 load_grid_from_xml 換成你的版本或直接餵資料即可。

import math, random
from typing import List, Tuple

import xml.etree.ElementTree as ET

def load(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'n': 'urn:cwa:gov:tw:cwacommon:0.1'}
    content = root.find('.//n:Resource/n:Content', ns).text

    # 同時處理逗號和空白
    raw_text = content.replace("\n"," ").replace("\r"," ")
    parts = raw_text.replace(",", " ").split()
    vals = [float(v) for v in parts if v.strip()!=""]

    # 格點規格
    lon0, lat0, d = 120.00, 21.88, 0.03
    nx, ny = 67, 120

    if len(vals) != nx*ny:
        raise ValueError(f"Expect {nx*ny} values, got {len(vals)}")

    X_all, temp_all = [], []
    for j in range(ny):
        for i in range(nx):
            lon = lon0 + d*i
            lat = lat0 + d*j
            t = vals[j*nx+i]
            X_all.append([lon, lat])
            temp_all.append(t)
    return X_all, temp_all



# ---------- 工具 ----------
def mean(xs): return sum(xs)/len(xs)
def stdev(xs):
    mu = mean(xs); return (sum((x-mu)**2 for x in xs)/len(xs))**0.5 or 1.0
def sigmoid(z):
    if z>=0: ez=math.exp(-z); return 1/(1+ez)
    ez=math.exp(z); return ez/(1+ez)

def fit_standardize(X: List[List[float]]):
    d=len(X[0]); mu=[mean([x[j] for x in X]) for j in range(d)]
    sd=[stdev([x[j] for x in X]) for j in range(d)]
    return mu, sd
def apply_standardize(X, mu, sd):
    return [[(x[j]-mu[j])/sd[j] for j in range(len(x))] for x in X]

# ---------- Logistic regression（交叉熵 + GD） ----------
def train_logreg(X, y, lr=0.2, epochs=2500, l2=1e-3, verbose=False):
    n,d=len(X),len(X[0]); w=[0.0]*d; b=0.0
    for ep in range(epochs):
        gw=[0.0]*d; gb=0.0; loss=0.0
        for i in range(n):
            z=sum(w[j]*X[i][j] for j in range(d))+b
            p=sigmoid(z); yi=y[i]
            diff=(p-yi)
            for j in range(d): gw[j]+=diff*X[i][j]
            gb+=diff
            eps=1e-12
            loss+=-(yi*math.log(p+eps)+(1-yi)*math.log(1-p+eps))
        for j in range(d): gw[j]=gw[j]/n + l2*w[j]
        gb/=n
        for j in range(d): w[j]-=lr*gw[j]
        b-=lr*gb
        if verbose and (ep%500==0 or ep==epochs-1):
            print(f"[logreg] ep={ep:4d} loss={loss/n:.4f}")
    return w,b

def prob_land(w,b,x): return sigmoid(sum(w[j]*x[j] for j in range(len(w)))+b)
def pred_land(w,b,x,thr=0.5): return 1 if prob_land(w,b,x)>=thr else 0

# ---------- Linear regression（常態方程 + 小 ridge） ----------
def solve_3x3(A,b):
    M=[A[0][:]+[b[0]],A[1][:]+[b[1]],A[2][:]+[b[2]]]
    for c in range(3):
        p=c
        for r in range(c,3):
            if abs(M[r][c])>abs(M[p][c]): p=r
        M[c],M[p]=M[p],M[c]
        if abs(M[c][c])<1e-12: raise ValueError("Singular")
        fac=M[c][c]
        for k in range(c,4): M[c][k]/=fac
        for r in range(c+1,3):
            f=M[r][c]
            for k in range(c,4): M[r][k]-=f*M[c][k]
    x=[0.0,0.0,0.0]
    for r in reversed(range(3)):
        s=M[r][3]-sum(M[r][k]*x[k] for k in range(r+1,3))
        x[r]=s/M[r][r]
    return x

def train_linreg(X, y, ridge=1e-6):
    n=len(X)
    S1=n
    Sx1=sum(x[0] for x in X); Sx2=sum(x[1] for x in X)
    Sy=sum(y)
    Sx1x1=sum(x[0]*x[0] for x in X); Sx2x2=sum(x[1]*x[1] for x in X)
    Sx1x2=sum(x[0]*x[1] for x in X)
    Sx1y=sum(x[0]*y[i] for i,x in enumerate(X))
    Sx2y=sum(x[1]*y[i] for i,x in enumerate(X))
    lam=ridge
    A=[[S1,  Sx1,          Sx2         ],
       [Sx1, Sx1x1+lam,    Sx1x2       ],
       [Sx2, Sx1x2,        Sx2x2+lam   ]]
    b=[Sy,Sx1y,Sx2y]
    return solve_3x3(A,b)  # [w0,w1,w2]

def predict_lin(w,x): return w[0]+w[1]*x[0]+w[2]*x[1]

# ---------- 合成新模型（門控回歸） ----------
class CombinedModel:
    def __init__(self, mu_c, sd_c, w_log, b_log, mu_r, sd_r, w_lin, thr=0.5):
        self.mu_c, self.sd_c = mu_c, sd_c
        self.w_log, self.b_log = w_log, b_log
        self.mu_r, self.sd_r = mu_r, sd_r
        self.w_lin = w_lin
        self.thr = thr

    def _z_c(self, x):  # x=[lon,lat]
        return [(x[j]-self.mu_c[j])/self.sd_c[j] for j in range(2)]
    def _z_r(self, x):
        return [(x[j]-self.mu_r[j])/self.sd_r[j] for j in range(2)]

    def prob(self, lon, lat):
        return prob_land(self.w_log, self.b_log, self._z_c([lon,lat]))

    def classify(self, lon, lat):
        return 1 if self.prob(lon,lat) >= self.thr else 0

    def regress(self, lon, lat):
        return predict_lin(self.w_lin, self._z_r([lon,lat]))

    def predict_hard(self, lon, lat):
        # 海上(0) → 0；陸地(1) → 回歸溫度
        return 0.0 if self.classify(lon,lat)==0 else self.regress(lon,lat)

    def predict_soft(self, lon, lat):
        # 概率加權，平滑邊界
        p = self.prob(lon,lat)
        return p * self.regress(lon,lat)

# ---------- 訓練流程樣板 ----------
def train_combined(X_all: List[List[float]], temp_all: List[float], invalid=-999.0, thr=0.5):
    """
    X_all: [[lon,lat], ...]
    temp_all: [t or -999, ...]
    回傳: CombinedModel
    """
    # 1) 分類資料（有效=1 / 無效(-999)=0）
    y_cls = [0 if t==invalid else 1 for t in temp_all]
    mu_c, sd_c = fit_standardize(X_all)
    Xc = apply_standardize(X_all, mu_c, sd_c)
    w_log, b_log = train_logreg(Xc, y_cls, lr=0.2, epochs=2500, l2=1e-3, verbose=False)

    # 2) 回歸資料（只保留有效）
    X_reg = [x for x,t in zip(X_all,temp_all) if t!=invalid]
    y_reg = [t for t in temp_all if t!=invalid]
    mu_r, sd_r = fit_standardize(X_reg)
    Xr = apply_standardize(X_reg, mu_r, sd_r)
    w_lin = train_linreg(Xr, y_reg, ridge=1e-6)

    # 3) 合成新模型
    return CombinedModel(mu_c, sd_c, w_log, b_log, mu_r, sd_r, w_lin, thr=thr)

# ====== 使用示例 ======
if __name__ == "__main__":
    # 使用你的 XML
    X_all, temp_all = load("C:/Users/User/Desktop/文件/O-A0038-003.xml")

    # 訓練合成模型
    model = train_combined(X_all, temp_all, invalid=-999.0, thr=0.5)

    # 輸出全部 nodes 的預測結果
    import csv
    with open("prediction_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lon","lat","p_land","class","T_hard","T_soft"])
        for (lon,lat) in X_all:
            writer.writerow([
                f"{lon:.3f}", f"{lat:.3f}",
                f"{model.prob(lon,lat):.4f}",
                model.classify(lon,lat),
                f"{model.predict_hard(lon,lat):.2f}",
                f"{model.predict_soft(lon,lat):.2f}"
            ])

    print("[+] 預測已完成，結果存到 prediction_output.csv")
