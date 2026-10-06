"""Задания 14-33: графики, статистика, спектральный анализ, интерполяция."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import convolve, periodogram, spectrogram


# --- Задание 14 ---------------------------------------------------------
def plot_series(data: np.ndarray, names=None, title: str = "Исходные данные"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=names[i] if names else f"col_{i}", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# --- Задание 15 ---------------------------------------------------------
def plot_histogram(x: np.ndarray, bins: int = 30, name: str = "x1"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black",
            label="гистограмма")
    grid = np.linspace(x.min(), x.max(), 400)
    ax.plot(grid, stats.gaussian_kde(x)(grid), "r-", lw=2,
            label="оценка плотности (KDE)")
    ax.set_title(f"Нормализованная гистограмма и плотность ({name})")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    ax.legend()
    return fig


# --- Задание 16 ---------------------------------------------------------
def show_sorted(df: pd.DataFrame, n: int = 5) -> None:
    for col in df.columns:
        s = np.sort(df[col].to_numpy())
        print(f"--- {col} (первые {n} / последние {n} из {len(s)}) ---")
        print("  min:", np.round(s[:n], 4), " max:", np.round(s[-n:], 4))


# --- Задание 17 ---------------------------------------------------------
def plot_ecdf(x: np.ndarray, name: str = "x1"):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(xs, ys, where="post")
    ax.set_title(f"Эмпирическая функция распределения ({name})")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.grid(True, alpha=0.3)
    return fig


# --- Задание 18 ---------------------------------------------------------
def column_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "mode": float(stats.mode(x, keepdims=False).mode),
        "median": float(np.median(x)),
    }


# --- Задание 19 ---------------------------------------------------------
def ci_mean(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)


def ci_var(x: np.ndarray, alpha: float = 0.05):
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low


# --- Задание 22 ---------------------------------------------------------
def cross_correlation(x, y, max_lags: int = 50):
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]


def plot_cross_correlation(lags, c, names=("x1", "x2")):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(lags, c, basefmt=" ")
    ax.set_title(f"Взаимная корреляция {names[0]} и {names[1]}")
    ax.set_xlabel("Сдвиг (lag)")
    ax.set_ylabel("Коэффициент")
    ax.grid(True, alpha=0.3)
    return fig


# --- Задание 23 ---------------------------------------------------------
def plot_gradient(x: np.ndarray, n: int = 100, name: str = "x1"):
    dx = np.gradient(x)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x[:n]); axes[0].set_title(f"{name}: исходный сигнал")
    axes[1].plot(dx[:n], color="tab:red"); axes[1].set_title(f"{name}: производная (np.gradient)")
    for a in axes:
        a.grid(True, alpha=0.3)
    axes[1].set_xlabel("Индекс")
    return fig


# --- Задания 28-30 ------------------------------------------------------
def plot_spectrogram(x: np.ndarray, name: str = "x1"):
    f, t, Sxx = spectrogram(x, fs=1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    m = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    fig.colorbar(m, ax=ax, label="дБ")
    ax.set_ylabel("Частота")
    ax.set_xlabel("Время")
    ax.set_title(f"Спектрограмма ({name})")
    return fig


def plot_periodogram(x: np.ndarray, name: str = "x1"):
    f, Pxx = periodogram(x, fs=1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f[1:], Pxx[1:])
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    ax.set_title(f"Периодограмма ({name})")
    ax.grid(True, alpha=0.3)
    return fig


def plot_fft_amplitude(x: np.ndarray, name: str = "x1"):
    n = len(x)
    fft = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freq[:half], np.abs(fft[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    ax.set_title(f"АЧХ через FFT ({name})")
    ax.grid(True, alpha=0.3)
    return fig


# --- Задания 31-32 ------------------------------------------------------
def interpolation_demo(x: np.ndarray, n_points: int = 30, name: str = "x1"):
    """Кубическая интерполяция (interp1d) и сплайн (UnivariateSpline)."""
    x = x[:n_points]
    idx = np.arange(len(x))
    x_new = np.linspace(0, len(x) - 1, 10 * len(x))
    y_cubic = interp1d(idx, x, kind="cubic", fill_value="extrapolate")(x_new)
    y_spl = UnivariateSpline(idx, x, k=3, s=0)(x_new)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(idx, x, "o", label="исходные точки")
    axes[0].plot(x_new, y_cubic, "-", label="кубическая интерполяция")
    axes[0].set_title(f"Кубическая интерполяция interp1d ({name})")
    axes[1].plot(idx, x, "o", label="исходные точки")
    axes[1].plot(x_new, y_spl, "-", color="tab:green", label="UnivariateSpline (k=3, s=0)")
    axes[1].set_title(f"Интерполяция сплайном ({name})")
    for a in axes:
        a.legend(); a.grid(True, alpha=0.3)
    axes[1].set_xlabel("Индекс")
    return fig, float(np.max(np.abs(y_cubic - y_spl)))


# --- Задание 33 ---------------------------------------------------------
def binary_masks(x: np.ndarray) -> dict:
    return {
        "x > 0": x > 0,
        "x < 0": x < 0,
        "x == 0": x == 0,
        "-1 <= x <= 1": (x >= -1) & (x <= 1),
    }
