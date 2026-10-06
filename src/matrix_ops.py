"""Задания 46-49: разреженные матрицы, обращение, трёхмерный тензор, PCA."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def sparse_demo():
    """Задание 46."""
    # Generator (а не глобальный np.random.seed) нужен, чтобы scipy не пытался
    # создавать перестановку из 10^10 элементов (~75 ГБ памяти)
    rng = np.random.default_rng(42)
    S = sparse.random(100_000, 100_000, density=1e-5, format="csr", random_state=rng)
    return S


def inverse_demo(size: int = 100):
    """Задание 47."""
    M = np.random.rand(size, size)
    det = np.linalg.det(M)
    if abs(det) > 1e-10:
        M_inv = np.linalg.inv(M)
        ok = bool(np.allclose(M @ M_inv, np.eye(size), atol=1e-6))
        err = float(np.max(np.abs(M @ M_inv - np.eye(size))))
        return dict(det=float(det), singular=False, ok=ok, err=err)
    return dict(det=float(det), singular=True, ok=False, err=None)


def build_F(X: np.ndarray) -> np.ndarray:
    """Задание 48. F формы (n, m, 3): исходные, MinMax, Standard."""
    X_mm = MinMaxScalerCustom().fit_transform(X)
    X_std = StandardScalerCustom().fit_transform(X)
    return np.stack([X, X_mm, X_std], axis=-1)


def _loglik(data: np.ndarray, n_components=None) -> dict:
    """Суммарное log-правдоподобие PCA-модели для двух вариантов масштабирования."""
    results = {}
    for name, scaler in {"minmax": MinMaxScaler(), "std": StandardScaler()}.items():
        pipe = Pipeline([("scaler", scaler), ("pca", PCA(n_components=n_components))])
        pipe.fit(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # плохо обусловленная ковариация
            ll = pipe.named_steps["pca"].score_samples(
                pipe.named_steps["scaler"].transform(data)).sum()
        results[name] = float(ll)
    return results


def pca_pipeline(F: np.ndarray, X: np.ndarray):
    """Задание 49. Сравнение MinMax и Standard через Pipeline(scaler, PCA).

    (а) по методичке: все 12 признаков F.reshape(n, -1). Три слоя F — линейные
        функции одних и тех же 4 столбцов, поэтому ковариационная матрица вырождена
        и для одного из скейлеров правдоподобие может оказаться -inf;
    (б) дополнительно: исходные 4 признака, PCA(n_components=3) (вероятностная
        модель с ненулевым шумом) — оба значения конечны.
    """
    flat = F.reshape(len(F), -1)
    res_full = _loglik(flat)
    res_x = _loglik(X, n_components=X.shape[1] - 1)
    finite = {k: v for k, v in res_full.items() if np.isfinite(v)}
    best_full = max(finite, key=finite.get) if finite else None
    best_x = max(res_x, key=res_x.get)

    pca = PCA().fit(flat)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")
    ax.set_title("PCA: кумулятивная объяснённая дисперсия")
    ax.grid(True, alpha=0.3)
    return dict(full=res_full, best_full=best_full, x4=res_x, best_x4=best_x), fig, pca
