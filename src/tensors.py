"""Задания 38-45: тензорные вычисления в TensorFlow, PyTorch и Keras."""
import numpy as np
import pandas as pd

from src.io_utils import write_table


def run_tensor_tasks(data: np.ndarray, k: int = 5, p: int = 4) -> dict:
    import keras
    import tensorflow as tf
    import torch
    from keras import ops

    res = {}

    # --- Задание 38: тензор TensorFlow --------------------------------
    X_tf = tf.constant(data, dtype=tf.float32)
    print(f"[38] X_tf: shape={tuple(X_tf.shape)}, dtype={X_tf.dtype.name}")
    print("     первые 2 строки:", X_tf[:2].numpy().round(4).tolist())

    # --- Задание 39: тензор PyTorch -----------------------------------
    X_pt = torch.tensor(data, dtype=torch.float32)
    print(f"[39] X_pt: shape={tuple(X_pt.shape)}, dtype={X_pt.dtype}")
    print("     первые 2 строки:", X_pt[:2].numpy().round(4).tolist())
    print("     X_tf == X_pt:", bool(np.allclose(X_tf.numpy(), X_pt.numpy())))

    # --- Задание 40: случайные тензоры --------------------------------
    n, m = X_tf.shape
    A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
    W = tf.random.normal((m, p))
    B = tf.random.uniform((k, p))
    print(f"[40] A: {tuple(A.shape)} {A.dtype.name} (int), W: {tuple(W.shape)} "
          f"(normal), B: {tuple(B.shape)} (uniform)")

    # --- Задание 41: AXW + B (TensorFlow) -----------------------------
    AX = tf.matmul(tf.cast(A, tf.float32), X_tf)
    AXW = tf.matmul(AX, W)
    result_tf = AXW + B
    print(f"[41] TensorFlow: A@X -> {tuple(AX.shape)}, A@X@W + B -> {tuple(result_tf.shape)}")
    print(result_tf.numpy().round(3))

    # --- Задание 42: AXW + B (PyTorch) --------------------------------
    A_pt = torch.randint(0, 10, (k, n)).to(torch.float32)
    W_pt = torch.randn(m, p)
    B_pt = torch.rand(k, p)
    result_pt = A_pt @ X_pt @ W_pt + B_pt
    print(f"[42] PyTorch: A@X@W + B -> {tuple(result_pt.shape)}")
    print(result_pt.numpy().round(3))

    # --- Задание 43: матрицы T, P, Q ----------------------------------
    T = np.random.rand(3, 10)
    P = np.random.rand(3, 10)
    Q = np.random.rand(3, 10)
    print(f"[43] T, P, Q: shape={T.shape}, значения в (0, 1)")

    # --- Задание 44: операция в Keras ---------------------------------
    T_k = ops.convert_to_tensor(T)
    P_k = ops.convert_to_tensor(P)
    Q_k = ops.convert_to_tensor(Q)
    V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))
    V_k_np = ops.convert_to_numpy(V_k)
    print(f"[44] Keras {keras.__version__}: V = |sin(T) - exp(P)*sqrt(Q)|, shape={V_k_np.shape}")
    print(V_k_np.round(4))

    # --- Задание 45: та же операция в PyTorch -------------------------
    T_t, P_t, Q_t = torch.tensor(T), torch.tensor(P), torch.tensor(Q)
    V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))
    print(f"[45] PyTorch: shape={tuple(V_t.shape)}")
    print("     Keras == PyTorch:", bool(np.allclose(V_k_np, V_t.numpy(), atol=1e-10)),
          f"(макс. расхождение {np.max(np.abs(V_k_np - V_t.numpy())):.2e})")

    # Сохраняем результаты в файлы
    paths = write_table(pd.DataFrame(result_tf.numpy(),
                                     columns=[f"c{i}" for i in range(p)]),
                        "task41_result_tf")
    paths2 = write_table(pd.DataFrame(V_k_np,
                                      columns=[f"c{i}" for i in range(10)]),
                         "task44_V_keras")
    print("     сохранено:", paths["csv"], paths2["csv"])

    res.update(result_tf=result_tf.numpy(), result_pt=result_pt.numpy(),
               V_keras=V_k_np, V_torch=V_t.numpy())
    return res
