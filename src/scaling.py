"""Задания 34-35: сравнение собственных скейлеров со scikit-learn и инверсия."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.io_utils import write_table
from src.preprocess import MinMaxScalerCustom, StandardScalerCustom


def compare_scalers(x: np.ndarray, name: str = "x1"):
    """Задание 34. x — столбец формы (n, 1)."""
    out = {}
    for label, custom, sk in (("MinMax", MinMaxScalerCustom(), MinMaxScaler()),
                              ("Standard", StandardScalerCustom(), StandardScaler())):
        x_custom = custom.fit_transform(x)
        x_sk = sk.fit_transform(x)
        ok = np.allclose(x_custom, x_sk, atol=1e-10)
        out[label] = dict(custom=custom, sk=sk, x_custom=x_custom, x_sk=x_sk,
                          allclose=bool(ok),
                          max_diff=float(np.max(np.abs(x_custom - x_sk))))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x[:100, 0]); axes[0].set_title(f"Исходный {name}")
    axes[1].plot(out["MinMax"]["x_custom"][:100, 0], label="свой")
    axes[1].plot(out["MinMax"]["x_sk"][:100, 0], "--", label="sklearn")
    axes[1].set_title("MinMax [0, 1]"); axes[1].legend()
    axes[2].plot(out["Standard"]["x_custom"][:100, 0], label="свой")
    axes[2].plot(out["Standard"]["x_sk"][:100, 0], "--", label="sklearn")
    axes[2].set_title("Standard (z-score)"); axes[2].legend()
    for a in axes:
        a.grid(True, alpha=0.3)
    table = pd.DataFrame({
        name: x[:, 0],
        "minmax_custom": out["MinMax"]["x_custom"][:, 0],
        "minmax_sklearn": out["MinMax"]["x_sk"][:, 0],
        "std_custom": out["Standard"]["x_custom"][:, 0],
        "std_sklearn": out["Standard"]["x_sk"][:, 0],
    })
    paths = write_table(table, "task34_scalers")
    return out, fig, paths


def check_inverse(x: np.ndarray, res: dict) -> dict:
    """Задание 35. Обратное преобразование для обоих скейлеров."""
    report = {}
    for label, r in res.items():
        back_custom = r["custom"].inverse_transform(r["x_custom"])
        back_sk = r["sk"].inverse_transform(r["x_sk"])
        report[label] = dict(
            custom_ok=bool(np.allclose(x, back_custom, atol=1e-10)),
            sk_ok=bool(np.allclose(x, back_sk, atol=1e-10)),
            custom_err=float(np.max(np.abs(x - back_custom))),
            sk_err=float(np.max(np.abs(x - back_sk))),
        )
    return report
