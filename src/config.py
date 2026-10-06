"""Конфигурация проекта: seed, пути, константы."""
import os
import random
from datetime import datetime

import numpy as np

SEED = 42
VARIANT = 1

DATA_DIR = "data"
OUTPUT_DIR = "output"
DATASET_PATH = os.path.join(DATA_DIR, "dataset_var1.csv")
IMAGE_PATH = os.path.join(DATA_DIR, "image.png")

# Столбцы, нужные по варианту 1
VARIANT_COLUMNS = ["x1", "x2", "x3", "y"]
NUMERIC_COLS = ["x1", "x2", "x3", "y"]
CATEGORY_VALUES = {"category": ["A", "B", "C"]}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Фиксирует seed для всех генераторов случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:          # режим --skip-tensors без TensorFlow
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timestamped_filename(base: str, ext: str) -> str:
    """Формирует имя файла с датой и временем."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"{base}_{ts}.{ext}")
