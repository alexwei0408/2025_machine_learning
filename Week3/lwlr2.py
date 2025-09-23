import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Target function (Runge)
# ------------------------------
def runge(x):
    return -50*x/ (1.0 + 25.0 * x**2)**2

# ------------------------------
# Locally Weighted Linear Regression
# ------------------------------
def lwlr_point(xq, X, y, tau=0.15):
    w = np.exp(- (X - xq)**2 / (2.0 * tau**2))   # weights
    Phi = np.hstack([np.ones_like(X), X])        # design matrix [1, x]
    A = (Phi * w).T @ Phi
    b = (Phi * w).T @ y
    theta = np.linalg.solve(A, b)
    xq_feat = np.array([1.0, xq])
    yq = xq_feat @ theta
    return yq.item()

def lwlr_predict(Xq, X, y, tau=0.15):
    return np.array([lwlr_point(xq, X, y, tau) for xq in Xq.ravel()]).reshape(-1,1)

# ------------------------------
# Chebyshev 第一類節點
# ------------------------------
def chebyshev_first_kind_nodes(n):
    k = np.arange(1, n+1)
    x = np.cos((2*k - 1) * np.pi / (2*n))
    return x.reshape(-1,1)

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # prediction grid
    Xg = np.linspace(-1, 1, 400).reshape(-1,1)
    yg = runge(Xg)

    tau = 0.1
    n_train = 150

    # Case 1: uniform random nodes
    X_uni = rng.uniform(-1, 1, size=(n_train, 1))
    y_uni = runge(X_uni)
    yhat_uni = lwlr_predict(Xg, X_uni, y_uni, tau=tau)

    # Case 2: Chebyshev nodes
    X_cheb = chebyshev_first_kind_nodes(n_train)
    y_cheb = runge(X_cheb)
    yhat_cheb = lwlr_predict(Xg, X_cheb, y_cheb, tau=tau)

    # Metrics
    mse_uni  = np.mean((yhat_uni - yg)**2)
    mse_cheb = np.mean((yhat_cheb - yg)**2)
    linf_uni  = np.max(np.abs(yhat_uni - yg))
    linf_cheb = np.max(np.abs(yhat_cheb - yg))

    print(f"[Uniform]   MSE={mse_uni:.3e},  L_inf={linf_uni:.3e}")
    print(f"[Chebyshev] MSE={mse_cheb:.3e}, L_inf={linf_cheb:.3e}")

    # Plot 1: Function vs Prediction
    plt.figure(figsize=(7,5))
    plt.plot(Xg, yg, 'k-', label="True f'(x)")
    plt.plot(Xg, yhat_uni, 'r-',
             label=f"LWLR (uniform) MSE={mse_uni:.2e}, L_inf={linf_uni:.2e}")
    plt.plot(Xg, yhat_cheb, 'b--',
             label=f"LWLR (chebyshev) MSE={mse_cheb:.2e}, L_inf={linf_cheb:.2e}")
    plt.scatter(X_uni, y_uni, s=10, alpha=0.3, color="red")
    plt.scatter(X_cheb, y_cheb, s=10, alpha=0.3, color="blue")
    plt.title(f"Derivative Aprroximation with LWLR\n(tau={tau}, n_train={n_train})")

    plt.xlabel("x"); plt.ylabel("y"); plt.legend(fontsize=8); plt.tight_layout()
    plt.show()

    # Plot 2: Absolute Error Comparison
    plt.figure(figsize=(7,5))
    plt.plot(Xg, np.abs(yhat_uni - yg), 'r-',
             label=f"|error| (uniform) L_inf={linf_uni:.2e}")
    plt.plot(Xg, np.abs(yhat_cheb - yg), 'b--',
             label=f"|error| (chebyshev) L_inf={linf_cheb:.2e}")
    plt.title("Absolute Error of Derivative Approximation")
    plt.xlabel("x"); plt.ylabel("Absolute error"); plt.legend(fontsize=8); plt.tight_layout()
    plt.show()
