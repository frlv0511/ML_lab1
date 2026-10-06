import pandas as pd
import numpy as np
from src.data_loader import check_data_quality


def cast_types(df: pd.DataFrame, type_map: dict | str) -> pd.DataFrame:
    """Приводит тип самого столбца (не ячейки).

    type_map: {'col': 'float64', ...} или строка, например 'float'.
    """
    df = df.copy()
    if isinstance(type_map, str):
        df = df.astype(type_map)
    elif isinstance(type_map, dict):
        for col, dtype in type_map.items():
            df[col] = df[col].astype(dtype)
    return df


def drop_invalid(df: pd.DataFrame,
                 numeric_cols=None,
                 category_values=None,
                 date_cols=None) -> pd.DataFrame:
    """
    Удаляет строки с некорректными данными:
      • нечисловое значение (не NaN) в числовом столбце;
      • значение категории вне разрешённого набора;
      • дата, не парсящаяся в datetime.
    Пропуски (NaN) НЕ считаются некорректными и сохраняются.
    Побочный эффект: числовые столбцы и даты приводятся к правильным типам.
    """
    df = df.copy()

    # 1. Числовые столбцы
    if numeric_cols:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col]
            coerced = pd.to_numeric(s, errors="coerce")
            # errors="coerce" означает: если значение не преобразуется — поставить NaN, а не выбрасывать ошибку.
            bad = s.notna() & coerced.isna()
            df = df[~bad]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Категориальные столбцы
    if category_values:
        for col, allowed in category_values.items():
            if col not in df.columns:
                continue
            df = df[df[col].isna() | df[col].isin(allowed)]

    # 3. Даты
    if date_cols:
        for col in date_cols:
            if col not in df.columns:
                continue
            coerced = pd.to_datetime(df[col], errors="coerce")
            bad = df[col].notna() & coerced.isna()
            df = df[~bad]
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def handle_missing(df: pd.DataFrame,
                   action: str,
                   columns=None,
                   value=None,
                   dtype=None,
                   row=None,
                   col=None) -> pd.DataFrame:
    """
    Универсальная обработка пропусков и ячеек.

    action:
      'dropnan'  — удалить строки с NaN (в columns, если указано)
      'droptype' — удалить строки, где есть значение указанного типа (str/int/float/bool)
      'cast'     — привести значения к числу через pd.to_numeric(errors='coerce')
      'mean'     — заполнить средним по столбцу
      'ffill'    — заполнить предыдущим значением
      'bfill'    — заполнить следующим значением
      'zero'     — заполнить нулями
      'value'    — заполнить value (для columns) ИЛИ конкретную ячейку (row, col)
    """
    df = df.copy()

    if action == "dropnan":
        df = df.dropna(subset=columns)

    elif action == "droptype":
        if dtype == "str":
            fn = lambda x: isinstance(x, str)
        elif dtype == "int":
            fn = lambda x: isinstance(x, int) and not isinstance(x, bool)
        elif dtype == "float":
            fn = lambda x: isinstance(x, float)
        elif dtype == "bool":
            fn = lambda x: isinstance(x, bool)
        else:
            raise ValueError(f"Неизвестный dtype: {dtype}")
        mask = df.map(fn)
        if columns:
            mask = mask[columns]
        df = df[~mask.any(axis=1)]

    elif action == "cast":
        cols = columns if columns else df.columns
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    elif action == "mean":
        cols = columns if columns else df.select_dtypes(include="number").columns
        df[cols] = df[cols].fillna(df[cols].mean())

    elif action == "ffill":
        cols = columns if columns else df.columns
        df[cols] = df[cols].ffill()

    elif action == "bfill":
        cols = columns if columns else df.columns
        df[cols] = df[cols].bfill()

    elif action == "zero":
        cols = columns if columns else df.columns
        df[cols] = df[cols].fillna(0)

    elif action == "value":
        if row is not None and col is not None:
            df.at[row, col] = value          # конкретная ячейка
        else:
            cols = columns if columns else df.columns
            df[cols] = df[cols].fillna(value)  # залить все пропуски в столбцах

    else:
        raise ValueError(f"Неизвестное действие: {action}")

    return df


def interactive_menu(df: pd.DataFrame,
                     numeric_cols=None,
                     category_values=None) -> pd.DataFrame:
    while True:
        print("\n=== Меню обработки ===")
        print("1. Показать проблемы")
        print("2. Удалить строки с некорректными данными")
        print("3. Обработать пропуски в столбцах")
        print("4. Заполнить конкретную ячейку")
        print("5. Выход")

        choice = input("Выбор: ").strip()

        if choice == "1":
            print(check_data_quality(df, numeric_cols, category_values))

        elif choice == "2":
            df = drop_invalid(df, numeric_cols, category_values,
                              date_cols=["timestamp"])
            print(f"Готово. Осталось строк: {len(df)}")

        elif choice == "3":
            cols_in = input("Столбцы через запятую (Enter — все): ").strip()
            columns = [c.strip() for c in cols_in.split(",")] if cols_in else None
            action = input(
                "Действие (cast/mean/ffill/bfill/zero/value/dropnan): "
            ).strip()

            value = None
            if action == "value":
                raw = input("Значение: ").strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = raw

            df = handle_missing(df, action, columns=columns, value=value)
            print("Готово.")

        elif choice == "4":
            col_name = input("Столбец: ").strip()
            row_idx = int(input("Индекс строки: ").strip())
            raw = input("Значение: ").strip()
            try:
                val = float(raw)
            except ValueError:
                val = raw
            df = handle_missing(df, "value", row=row_idx, col=col_name, value=val)
            print("Готово.")

        elif choice == "5":
            break

    return df


def to_numpy(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float64)


class MinMaxScalerCustom:
    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a, self.b = a, b
        self.min_, self.max_ = None, None

    def _fit(self, x: np.ndarray):
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return self.a + (x - self.min_) * (self.b - self.a) / denom

    def fit_transform(self, x: np.ndarray) -> np.ndarray: 
        return self._fit(x)._transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        denom = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return (x_new - self.a) * denom / (self.b - self.a) + self.min_


class StandardScalerCustom:
    def __init__(self):
        self.mean_, self.std_ = None, None

    def _fit(self, x: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def _transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return (x - self.mean_) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self._fit(x)._transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        std = np.where(self.std_ == 0, 1, self.std_)
        return x_new * std + self.mean_


def split_train_val_test(x: np.ndarray, ratios, percent: bool = False):
    """ratios: (r1, r2, r3). percent=True — в процентах."""
    r = np.array(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_test = n - n_train - n_val

    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    return x_train, x_val, x_test