"""Задания 36-37: скользящее окно и скользящее среднее."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """Матрица окон формы (n - width + 1, width)."""
    if width <= 0 or width > len(x):
        raise ValueError("Некорректная ширина окна")
    n = len(x)
    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    return sliding_window(x, width).mean(axis=1)


def plot_moving_average(x: np.ndarray, width: int, n: int = 100, name: str = "x1"):
    ma = moving_average(x, width)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x[:n], alpha=0.6, label=f"{name}")
    ax.plot(np.arange(width - 1, width - 1 + len(ma))[: n - width + 1],
            ma[: n - width + 1], lw=2, label=f"скользящее среднее (окно {width})")
    ax.set_title("Скользящее среднее")
    ax.set_xlabel("Индекс")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
