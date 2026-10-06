import pandas as pd


def check_data_quality(df: pd.DataFrame,
                       numeric_cols=None,
                       category_values=None) -> dict:
    """Проверяет пропуски, типы, нечисловые значения, битые даты и категории."""
    report = {
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {},
        "invalid_categories": {},
        "invalid_dates": {},
    }

    # 1. Числовые столбцы: непустое значение не парсится в число
    cols = numeric_cols if numeric_cols is not None else df.columns
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        coerced = pd.to_numeric(s, errors="coerce")
        # errors="coerce" означает: если значение не преобразуется — поставить NaN, а не выбрасывать ошибку.
        is_null = s.isna()
        bad = s[~is_null & coerced.isna()]
        if not bad.empty:
            report["non_numeric"][col] = bad.tolist()[:10]

    # 2. Категориальные столбцы: значения вне допустимого набора
    if category_values:
        for col, allowed in category_values.items():
            if col not in df.columns:
                continue
            bad = df.loc[df[col].notna() & ~df[col].isin(allowed), col]
            if not bad.empty:
                report["invalid_categories"][col] = bad.tolist()[:10]

    # 3. Даты: не парсятся в datetime
    for col in df.columns:
        if col == "timestamp" or "date" in col.lower() or "time" in col.lower():
            coerced = pd.to_datetime(df[col], errors="coerce")
            bad = df.loc[df[col].notna() & coerced.isna(), col]
            if not bad.empty:
                report["invalid_dates"][col] = bad.tolist()[:10]

    return report


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")


def extract_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Возвращает новый DataFrame с нужными столбцами."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет столбцов: {missing}")
    return df[cols].copy()
