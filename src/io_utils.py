"""Задания 1-3: чтение и запись данных, сохранение изображений."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from src.config import timestamped_filename


def read_table(path: str) -> pd.DataFrame:
    """Задание 1. Универсальное чтение табличных данных (xlsx/csv/txt/mat)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(p)
    elif ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".txt":
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    elif ext == ".mat":
        mat = loadmat(p)
        arrays = {k: v for k, v in mat.items() if not k.startswith("__")}
        key = next(iter(arrays))
        df = pd.DataFrame(arrays[key])
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    # Автоматическое именование столбцов, если заголовков не было
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def write_table(df: pd.DataFrame, base_name: str) -> dict:
    """Задание 2. Сохраняет DataFrame во все форматы с timestamp в имени."""
    paths = {}
    paths["xlsx"] = timestamped_filename(base_name, "xlsx")
    df.to_excel(paths["xlsx"], index=False, engine="openpyxl")
    paths["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(paths["csv"], index=False)
    paths["txt"] = timestamped_filename(base_name, "txt")
    df.to_csv(paths["txt"], sep="\t", index=False)
    paths["mat"] = timestamped_filename(base_name, "mat")
    savemat(paths["mat"], {"data": df.to_numpy(),
                           "columns": np.array(df.columns, dtype=object)})
    return paths


def save_figure(fig, base_name: str) -> str:
    """Задание 3. Сохраняет рисунок в PNG с timestamp в имени."""
    path = timestamped_filename(base_name, "png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
