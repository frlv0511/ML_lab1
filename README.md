# Лабораторная работа №1. Библиотеки Python для машинного обучения. Работа с данными

Дисциплина «Машинное обучение», ДГТУ, группа ИПБП41. Студент: Фролов Е.С. **Вариант 1** (файл `data/dataset_var1.csv`, столбцы `x1, x2, x3, y`).

## Установка

```bash
python -m venv lab1_env
# Windows (PowerShell): .\lab1_env\Scripts\Activate.ps1      Linux/macOS: source lab1_env/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python main.py                 # все задания 1-50 (меню обработки данных не вызывается, пропуски заполняются средним)
python main.py --interactive   # то же, но задание 8 выполняется через интерактивное меню
python main.py --skip-tensors  # без заданий 38-45 (TensorFlow / PyTorch / Keras)
python main.py --only-tensors  # только задания 38-45
```

Все результаты (графики `*.png`, таблицы `*.xlsx/*.csv/*.txt/*.mat`, журнал выполнения `run_log_*.txt`) сохраняются в `output/` с меткой даты и времени в имени.

## Структура

| Файл | Задания |
|---|---|
| `main.py` | точка входа, последовательный запуск заданий |
| `src/config.py` | seed, пути, константы, `timestamped_filename` |
| `src/io_utils.py` | 1-3: чтение/запись xlsx, csv, txt, mat; сохранение рисунков |
| `src/data_loader.py` | 4-7: проверка данных, вывод таблицы, извлечение столбцов (файл преподавателя) |
| `src/preprocess.py` | 7-13: типы, меню обработки, NumPy, скейлеры, разбиение (файл преподавателя) |
| `src/stats_analysis.py` | 14-33: графики, статистики, доверительные интервалы, спектры, интерполяция, маски |
| `src/scaling.py` | 34-35: сравнение со scikit-learn, инверсия |
| `src/windows.py` | 36-37: скользящее окно и среднее |
| `src/tensors.py` | 38-45: тензоры TensorFlow, PyTorch, Keras |
| `src/matrix_ops.py` | 46-49: разреженные матрицы, обращение, тензор F, PCA-пайплайн |
| `src/image_proc.py` | 50: обработка изображения (10 операций) |

Для задания 50 использовано изображение `data/image.png` (`skimage.data.astronaut`, общественное достояние, NASA).
