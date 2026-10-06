"""Лабораторная работа №1. Библиотеки Python для машинного обучения.
Работа с данными. Вариант 1 (столбцы x1, x2, x3, y).

Запуск:
    python main.py                    # все задания 1-50
    python main.py --interactive      # меню обработки данных (задание 8) вручную
    python main.py --skip-tensors     # без заданий 38-45 (нужны TF/PyTorch/Keras)
    python main.py --only-tensors     # только задания 38-45
"""
import argparse
import builtins
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.config import set_seeds, timestamped_filename
from src.data_loader import check_data_quality, extract_columns, show_table
from src.io_utils import read_table, save_figure, write_table
from src.preprocess import (MinMaxScalerCustom, StandardScalerCustom, cast_types,
                            handle_missing, interactive_menu, split_train_val_test,
                            to_numpy)


class Tee:
    """Дублирует вывод консоли в файл журнала (он сохраняется в output/)."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def header(num, title):
    print(f"\n=== Задание {num}. {title} ===")


def prepare_data(args):
    """Задания 1-13: чтение, проверка, предобработка. Возвращает df (x1,x2,x3,y)."""
    cols = config.VARIANT_COLUMNS

    header(1, "Чтение файла")
    df = read_table(config.DATASET_PATH)
    print(f"Файл {config.DATASET_PATH} прочитан: {df.shape[0]} строк, {df.shape[1]} столбцов")

    header(4, "Проверка данных")
    report = check_data_quality(df, config.NUMERIC_COLS, config.CATEGORY_VALUES)
    print("Пропуски:", {k: v for k, v in report["missing"].items() if v})
    print("Нечисловые значения:", report["non_numeric"] or "нет")
    print("Недопустимые категории:", report["invalid_categories"] or "нет")
    print("Некорректные даты:", report["invalid_dates"] or "нет")
    for col, n in report["missing"].items():
        if n:
            print(f"ПРЕДУПРЕЖДЕНИЕ: в столбце {col} {n} пустых ячеек")

    header(5, "Вывод таблицы")
    show_table(df, 5)

    header(8, "Меню обработки проблем")
    if args.interactive:
        df = interactive_menu(df, config.NUMERIC_COLS, config.CATEGORY_VALUES)
    else:
        df = handle_missing(df, "mean", columns=["x2"])
        print("Пропуски в x2 заполнены средним значением (handle_missing, action='mean')")
    print("Пропусков после обработки:", int(df[cols].isna().sum().sum()))

    header(6, "Извлечение столбцов по варианту")
    df = extract_columns(df, cols)
    print("Столбцы варианта 1:", list(df.columns), "| shape =", df.shape)

    header(7, "Явные типы данных")
    df = cast_types(df, {c: "float64" for c in cols})
    print(df.dtypes.to_string())

    header(2, "Запись файлов (xlsx, csv, txt, mat)")
    paths = write_table(df, "dataset_clean")
    for ext, p in paths.items():
        print(f"  {ext}: {p}")
    print("Проверка чтения обратно (задание 1):")
    for ext, p in paths.items():
        back = read_table(p)
        print(f"  {ext}: shape={back.shape}")

    header(3, "Сохранение изображений")
    print("Функция save_figure() сохраняет каждый график в output/ с меткой времени (dpi=150)")
    return df


def run_data_tasks(df):
    cols = list(df.columns)
    header(9, "DataFrame -> NumPy")
    data = to_numpy(df)
    print(f"data: shape={data.shape}, dtype={data.dtype}")
    x1, x2, x3, y = (data[:, i] for i in range(4))

    header(10, "Собственные скейлеры")
    mm = MinMaxScalerCustom(0, 1).fit_transform(data)
    sd = StandardScalerCustom().fit_transform(data)
    print("MinMax [0,1]: min =", mm.min(axis=0).round(3), "max =", mm.max(axis=0).round(3))
    print("Standard: mean =", sd.mean(axis=0).round(6), "std =", sd.std(axis=0).round(6))

    header(11, "Восстановление масштаба")
    sc = MinMaxScalerCustom(0, 1)
    back = sc.inverse_transform(sc.fit_transform(data))
    sc2 = StandardScalerCustom()
    back2 = sc2.inverse_transform(sc2.fit_transform(data))
    print("MinMax: max|x - inverse| =", f"{np.abs(back - data).max():.2e}")
    print("Standard: max|x - inverse| =", f"{np.abs(back2 - data).max():.2e}")

    header(12, "Разбиение на train/val/test")
    tr, va, te = split_train_val_test(data, (70, 15, 15), percent=True)
    print(f"Соотношение 70/15/15: train={tr.shape}, val={va.shape}, test={te.shape}")
    tr, va, te = split_train_val_test(data, (3, 1, 1))
    print(f"Соотношение 3:1:1:    train={tr.shape}, val={va.shape}, test={te.shape}")

    header(13, "Фиксация seed")
    print(f"set_seeds({config.SEED}) вызван в начале main.py")
    return data


def run_stats_tasks(df, data):
    from src import stats_analysis as sa
    from src import scaling, windows
    from scipy import stats
    import matplotlib.pyplot as plt

    names = list(df.columns)
    x1, x2, x3, y = (data[:, i] for i in range(4))

    header(14, "График исходных данных")
    fig = sa.plot_series(data, names)
    print("Сохранён:", save_figure(fig, "task14_series")); plt.close(fig)

    header(15, "Гистограмма и плотность (x1)")
    fig = sa.plot_histogram(x1)
    print("Сохранён:", save_figure(fig, "task15_hist_x1")); plt.close(fig)

    header(16, "Отсортированные столбцы")
    sa.show_sorted(df)

    header(17, "Эмпирическая функция распределения (x1)")
    fig = sa.plot_ecdf(x1)
    print("Сохранён:", save_figure(fig, "task17_ecdf_x1")); plt.close(fig)

    header(18, "Статистики")
    rows = {c: sa.column_stats(df[c].to_numpy()) for c in names}
    st = pd.DataFrame(rows).T
    print(st.round(4).to_string())
    print("Сверка с pandas: describe()")
    print(df.describe().round(4).to_string())
    write_table(st.reset_index().rename(columns={"index": "column"}), "task18_stats")

    header(19, "Доверительные интервалы (alpha=0.05)")
    ci_rows = []
    for c in names:
        v = df[c].to_numpy()
        lo, hi = sa.ci_mean(v); vlo, vhi = sa.ci_var(v)
        ci_rows.append(dict(column=c, mean_lo=lo, mean_hi=hi, var_lo=vlo, var_hi=vhi))
        print(f"{c}: среднее ({lo:.4f}; {hi:.4f}), дисперсия ({vlo:.4f}; {vhi:.4f})")
    write_table(pd.DataFrame(ci_rows), "task19_confidence_intervals")

    header(20, "Ковариация и корреляция")
    print("Ковариационная матрица (np.cov):")
    print(pd.DataFrame(np.cov(data, rowvar=False), index=names, columns=names).round(4).to_string())
    print("Корреляционная матрица (pandas .corr()):")
    print(df.corr().round(4).to_string())

    header(21, "Значимость коэффициента корреляции")
    for a, b, xa, xb in (("x1", "x2", x1, x2), ("x1", "y", x1, y), ("x3", "y", x3, y)):
        r, p = stats.pearsonr(xa, xb)
        verdict = "значима" if p < 0.05 else "незначима"
        print(f"corr({a},{b}): r={r:.4f}, p={p:.4g} -> корреляция {verdict}")

    header(22, "Взаимная корреляция (x1, x2)")
    lags, c = sa.cross_correlation(x1, x2)
    print(f"Максимум |c| = {np.abs(c).max():.4f} при lag = {lags[np.abs(c).argmax()]}")
    fig = sa.plot_cross_correlation(lags, c)
    print("Сохранён:", save_figure(fig, "task22_cross_corr")); plt.close(fig)

    header(23, "Производная и градиент")
    dx = np.gradient(x1)
    dx2 = np.gradient(data, axis=0)
    print("np.gradient(x1)[:5] =", dx[:5].round(4))
    print("np.gradient(data, axis=0): shape =", dx2.shape)
    fig = sa.plot_gradient(x1)
    print("Сохранён:", save_figure(fig, "task23_gradient")); plt.close(fig)

    header(24, "Свёртка (x1 * x2)")
    conv = np.convolve(x1, x2, mode="full")
    conv2 = stats and __import__("scipy.signal", fromlist=["convolve"]).convolve(x1, x2, mode="full")
    print(f"len(conv) = {len(conv)} (= {len(x1)} + {len(x2)} - 1); np.convolve == scipy: {np.allclose(conv, conv2)}")
    print("conv[:5] =", conv[:5].round(4))

    header(25, "Скалярное и векторное произведения")
    print("dot(x1, x2) =", round(float(np.dot(x1, x2)), 4))
    print("cross(x1[:3], x2[:3]) =", np.cross(x1[:3], x2[:3]).round(4))

    header(26, "Нормы L1 и L2 (x1)")
    print(f"L1 = {np.linalg.norm(x1, ord=1):.4f}, L2 = {np.linalg.norm(x1, ord=2):.4f}")

    header(27, "Проверка гипотез о распределении")
    for nm, v in (("x1", x1), ("x2", x2)):
        _, p_u = stats.kstest(v, "uniform", args=(v.min(), v.max() - v.min()))
        _, p_n = stats.kstest(v, "norm", args=(v.mean(), v.std()))
        _, p_s = stats.shapiro(v)
        print(f"{nm}: KS-равномерное p={p_u:.4g} | KS-нормальное p={p_n:.4g} | Шапиро p={p_s:.4g}")
        print(f"    равномерное: {'не отвергаем' if p_u > 0.05 else 'отвергаем'}; "
              f"нормальное (KS): {'не отвергаем' if p_n > 0.05 else 'отвергаем'}; "
              f"нормальное (Шапиро): {'не отвергаем' if p_s > 0.05 else 'отвергаем'}")

    header(28, "Спектрограмма (x1)")
    fig = sa.plot_spectrogram(x1)
    print("Сохранён:", save_figure(fig, "task28_spectrogram")); plt.close(fig)

    header(29, "Периодограмма (x1)")
    fig = sa.plot_periodogram(x1)
    print("Сохранён:", save_figure(fig, "task29_periodogram")); plt.close(fig)

    header(30, "АЧХ через FFT (x1)")
    fig = sa.plot_fft_amplitude(x1)
    print("Сохранён:", save_figure(fig, "task30_fft")); plt.close(fig)

    header(31, "Кубическая интерполяция и 32. Сплайны")
    fig, diff = sa.interpolation_demo(x1)
    print(f"Интерполяция по первым 30 точкам x1; макс. расхождение interp1d(cubic) и UnivariateSpline(s=0): {diff:.3e}")
    print("Сохранён:", save_figure(fig, "task31_32_interpolation")); plt.close(fig)

    header(33, "Бинарные маски (x1)")
    for k, m in sa.binary_masks(x1).items():
        print(f"  {k:14s}: {int(m.sum())} элементов ({m.mean() * 100:.1f} %)")

    header(34, "Сравнение собственного и sklearn-масштабирования")
    x = data[:, [0]]
    res, fig, paths = scaling.compare_scalers(x)
    for label, r in res.items():
        print(f"{label}: allclose(custom, sklearn, atol=1e-10) = {r['allclose']}, макс. расхождение = {r['max_diff']:.2e}")
    print("Сохранён:", save_figure(fig, "task34_scalers_plot")); plt.close(fig)
    print("Таблица результатов сохранена:", paths["csv"], "(+ xlsx, txt, mat)")

    header(35, "Инверсия масштабирования")
    for label, r in scaling.check_inverse(x, res).items():
        print(f"{label}: свой inverse OK={r['custom_ok']} (err={r['custom_err']:.2e}); "
              f"sklearn inverse OK={r['sk_ok']} (err={r['sk_err']:.2e})")

    header(36, "Скользящее окно (x1, ширина 5)")
    W = windows.sliding_window(x1, 5)
    print(f"Матрица окон: shape = {W.shape}")
    print("Первые 3 окна:\n", W[:3].round(4))

    header(37, "Скользящее среднее (x1, ширина 5)")
    ma = windows.moving_average(x1, 5)
    print(f"len(MA) = {len(ma)}, первые значения: {ma[:5].round(4)}")
    fig = windows.plot_moving_average(x1, 5)
    print("Сохранён:", save_figure(fig, "task37_moving_average")); plt.close(fig)


def run_matrix_tasks(data):
    from src import matrix_ops as mo
    import matplotlib.pyplot as plt

    header(46, "Разреженный массив SciPy")
    S = mo.sparse_demo()
    print(f"CSR-матрица {S.shape}, NNZ = {S.nnz}, формат = {S.format}")

    header(47, "Обратная матрица")
    r = mo.inverse_demo()
    print(f"det(M) = {r['det']:.4e}")
    print("Матрица вырождена" if r["singular"] else
          f"M @ M^-1 == E (atol=1e-6): {r['ok']}, макс. отклонение {r['err']:.2e}")

    header(48, "Достроить массив до F")
    F = mo.build_F(data)
    print("F.shape =", F.shape, "(n, m, 3): исходные | MinMax | Standard")

    header(49, "Пайплайн sklearn (PCA)")
    results, fig, pca = mo.pca_pipeline(F, data)
    print("(а) по методичке, F.reshape(n, -1) -> 12 признаков, PCA() без ограничения числа компонент:")
    for k, v in results["full"].items():
        print(f"    {k}: log-likelihood = {v:.4f}")
    print("    Лучший вариант (по конечным значениям):", results["best_full"])
    print("    (-inf означает вырожденную ковариацию: слои F линейно зависимы)")
    print("(б) исходные 4 признака, PCA(n_components=3):")
    for k, v in results["x4"].items():
        print(f"    {k}: log-likelihood = {v:.4f}")
    print("    Лучший вариант:", results["best_x4"])
    print("Кумулятивная объяснённая дисперсия (первые 5 компонент):",
          np.cumsum(pca.explained_variance_ratio_)[:5].round(6))
    print("Сохранён:", save_figure(fig, "task49_pca_variance")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true", help="интерактивное меню (задание 8)")
    ap.add_argument("--skip-tensors", action="store_true", help="пропустить задания 38-45")
    ap.add_argument("--only-tensors", action="store_true", help="выполнить только задания 38-45")
    args = ap.parse_args()

    # Если ввод идёт из файла/канала, повторяем введённые ответы в консоли
    if not sys.stdin.isatty():
        _input = builtins.input

        def echo_input(prompt=""):
            s = _input(prompt)
            print(s)
            return s
        builtins.input = echo_input

    suffix = "tensors" if args.only_tensors else "main"
    log_path = timestamped_filename(f"run_log_{suffix}", "txt")
    log = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    set_seeds(config.SEED)
    print(f"Лабораторная работа №1, вариант {config.VARIANT}. Запуск: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}")

    if args.only_tensors:
        df = prepare_quiet()
        from src.tensors import run_tensor_tasks
        header("38-45", "Тензорные вычисления")
        run_tensor_tasks(df.to_numpy(dtype=np.float64))
    else:
        df = prepare_data(args)
        data = run_data_tasks(df)
        run_stats_tasks(df, data)
        if not args.skip_tensors:
            from src.tensors import run_tensor_tasks
            header("38-45", "Тензорные вычисления")
            run_tensor_tasks(data)
        run_matrix_tasks(data)
        header(50, "Обработка изображения")
        from src.image_proc import run_task50
        run_task50(config.IMAGE_PATH)

    print(f"\nГотово. Журнал выполнения: {log_path}")
    sys.stdout = sys.__stdout__
    log.close()


def prepare_quiet():
    """Подготовка данных без вывода (для режима --only-tensors)."""
    df = read_table(config.DATASET_PATH)
    df = handle_missing(df, "mean", columns=["x2"])
    from src.preprocess import drop_invalid
    df = drop_invalid(df, config.NUMERIC_COLS, config.CATEGORY_VALUES, date_cols=["timestamp"])
    df = extract_columns(df, config.VARIANT_COLUMNS)
    return cast_types(df, {c: "float64" for c in config.VARIANT_COLUMNS})


if __name__ == "__main__":
    main()
