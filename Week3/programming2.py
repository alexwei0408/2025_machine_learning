
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

rng = np.random.default_rng(42)

def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def fprime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# ------------------------------
# Data
# ------------------------------
# Train on [-1, 1], with separate validation grid
N_train = 256
N_val = 256
x_train = np.linspace(-1, 1, N_train).reshape(-1, 1)
x_val = np.linspace(-1, 1, N_val).reshape(-1, 1)

y_train = f(x_train)
dy_train = fprime(x_train)

y_val = f(x_val)
dy_val = fprime(x_val)

m = 40  # hidden width (can increase if underfitting)

# Parameters (column vectors)
a0 = np.array([0.0])  # scalar bias
v = rng.normal(0, 0.1, size=(m, 1))
w = rng.normal(0, 1.0, size=(m, 1))
b = rng.normal(0, 0.1, size=(m, 1))

def forward(x):
    # x: (N,1)
    # Returns y, dy, plus cached intermediates for backprop
    z = w.T * x + b.T            # (N,m): z_ij = w_j*x_i + b_j
    t = np.tanh(z)               # (N,m)
    s = 1.0 - t**2               # sech^2(z) = 1 - tanh^2(z)
    y = a0 + (t @ v)             # (N,1)
    dy = (s * w.T) @ v           # (N,1) since each term v_j*s_ij*w_j
    cache = (x, z, t, s)
    return y, dy, cache

# ------------------------------
# Loss: L = λf * MSE(y, y*) + λd * MSE(y', y'*)
# ------------------------------
lam_f = 1.0
lam_d = 1.0

def loss_and_grads(x, y_true, dy_true):
    global a0, v, w, b
    N = x.shape[0]
    y_pred, dy_pred, (x_, z, t, s) = forward(x)
    r = y_pred - y_true   # (N,1)
    q = dy_pred - dy_true # (N,1)

    Lf = np.mean(r**2)
    Ld = np.mean(q**2)
    L = lam_f * Lf + lam_d * Ld

    # Precompute sums for grads
    # Shapes: t,s,z are (N,m); r,q (N,1)
    # For broadcasting correctness, convert to 2D
    rT = r.reshape(N,1)
    qT = q.reshape(N,1)

    # Common weighted sums
    # ∂L/∂a0
    da0 = (2.0 * lam_f / N) * np.sum(r)

    # For each hidden unit j we need sums over i
    # Build matrices helpful for vectorization
    # (N,m) elementwise products with r, q
    R = rT * np.ones_like(t)  # (N,m)
    Q = qT * np.ones_like(t)  # (N,m)

    # Grad wrt v: (m,1)
    # ∂L/∂v_j = (2λf/N) Σ_i r_i * t_ij + (2λd/N) Σ_i q_i * s_ij * w_j
    dv = (2.0 * lam_f / N) * (t.T @ r) + (2.0 * lam_d / N) * ((s.T @ q) * w)

    # Grad wrt w: (m,1)
    # ∂y_i/∂w_j = v_j * s_ij * x_i
    # ∂dy_i/∂w_j = v_j * (-2 t_ij s_ij) * x_i * w_j + v_j * s_ij
    xs = x_ @ np.ones((1, m))          # (N,m), repeats x along columns
    term_y_w = v.T * s * xs            # (N,m)
    term_dy_w = v.T * (-2.0 * t * s) * xs * (w.T) + v.T * s  # (N,m)
    dw = (2.0 * lam_f / N) * np.sum(R * term_y_w, axis=0, keepdims=True).T \
       + (2.0 * lam_d / N) * np.sum(Q * term_dy_w, axis=0, keepdims=True).T

    # Grad wrt b: (m,1)
    # ∂y_i/∂b_j = v_j * s_ij
    # ∂dy_i/∂b_j = v_j * (-2 t_ij s_ij) * w_j
    term_y_b = v.T * s
    term_dy_b = v.T * (-2.0 * t * s) * (w.T)
    db = (2.0 * lam_f / N) * np.sum(R * term_y_b, axis=0, keepdims=True).T \
       + (2.0 * lam_d / N) * np.sum(Q * term_dy_b, axis=0, keepdims=True).T

    return L, Lf, Ld, da0, dv, dw, db

# ------------------------------
# Adam optimizer
# ------------------------------
class Adam:
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.params = params

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

# Initialize optimizer
optimizer = Adam(params=[a0, v, w, b], lr=5e-3)

# ------------------------------
# Training loop
# ------------------------------
epochs = 4000
train_hist = []
val_hist = []

for ep in range(1, epochs + 1):
    L, Lf, Ld, da0, dv, dw, db = loss_and_grads(x_train, y_train, dy_train)
    optimizer.step([np.array([da0]), dv, dw, db])

    # Validation
    with np.errstate(all='ignore'):
        yv, dyv, _ = forward(x_val)
    rv = yv - y_val
    qv = dyv - dy_val
    Lfv = np.mean(rv**2)
    Ldv = np.mean(qv**2)
    Lv = lam_f * Lfv + lam_d * Ldv

    train_hist.append((Lf, Ld, L))
    val_hist.append((Lfv, Ldv, Lv))

# ------------------------------
# Plots
# ------------------------------
train_hist = np.array(train_hist)
val_hist = np.array(val_hist)

# Print final metrics
print("Final MSE (train): f =", float(train_hist[-1,0]), ", f' =", float(train_hist[-1,1]), ", total =", float(train_hist[-1,2]))
print("Final MSE (val):   f =", float(val_hist[-1,0]),   ", f' =", float(val_hist[-1,1]),   ", total =", float(val_hist[-1,2]))

# 1) Total loss curves
plt.figure()
plt.plot(train_hist[:,2], label="train total")
plt.plot(val_hist[:,2], label="val total")
plt.yscale('log')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Total loss (log scale)")
plt.legend()
plt.show()

# 2) Function vs. prediction
x_plot = np.linspace(-1, 1, 800).reshape(-1,1)
y_pred, dy_pred, _ = forward(x_plot)

plt.figure()
plt.plot(x_plot, f(x_plot), label="f(x) true")
plt.plot(x_plot, y_pred, label="NN f(x)")
plt.scatter(x_train, y_train, s=8, alpha=0.3, label="train pts")
plt.xlabel("x")
plt.ylabel("f")
plt.title("Runge function approximation")
plt.legend()
plt.show()

# 3) Derivative vs. prediction
plt.figure()
plt.plot(x_plot, fprime(x_plot), label="f'(x) true")
plt.plot(x_plot, dy_pred, label="NN f'(x)")
plt.scatter(x_train, fprime(x_train), s=8, alpha=0.3, label="train pts")
plt.xlabel("x")
plt.ylabel("f'")
plt.title("Runge derivative approximation")
plt.legend()
plt.show()

# 4) Separate loss components
plt.figure()
plt.plot(train_hist[:,0], label="train f-loss")
plt.plot(val_hist[:,0], label="val f-loss")
plt.yscale('log')
plt.xlabel("epoch")
plt.ylabel("MSE(f)")
plt.title("Function value loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(train_hist[:,1], label="train f'-loss")
plt.plot(val_hist[:,1], label="val f'-loss")
plt.yscale('log')
plt.xlabel("epoch")
plt.ylabel("MSE(f')")
plt.title("Derivative loss")
plt.legend()
plt.show()
