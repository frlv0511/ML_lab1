"""Задание 50. Обработка изображения средствами SciPy (10 операций)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

from src.io_utils import save_figure


def _before_after(img, new, title, base, cmap=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img); axes[0].set_title("Исходное")
    axes[1].imshow(new, cmap=cmap); axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    path = save_figure(fig, base)
    plt.close(fig)
    return path


# 1. Разделение каналов
def split_channels(img):
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение (H, W, 3)")
    r, g, b = (img[..., i].copy() for i in range(3))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);                axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds");     axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens");   axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues");    axes[3].set_title("Синий канал")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_1_channels")
    plt.close(fig)
    return {"r": r, "g": g, "b": b}


# 2. Поворот на 40 градусов
def rotate_image(img, angle: float = 40.0):
    rotated = rotate(img.astype(np.float32), angle=angle, reshape=True,
                     order=3, mode="constant", cval=0)
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    _before_after(img, rotated, f"Поворот на {angle}°", "task50_2_rotated")
    return rotated


# 3. Чёрно-белое изображение
def to_grayscale(img):
    gray_u8 = (rgb2gray(img) * 255).astype(np.uint8)
    _before_after(img, gray_u8, "Ч/б", "task50_3_gray", cmap="gray")
    return gray_u8


# 4. Гистограмма ч/б изображения
def gray_histogram(gray, bins: int = 256):
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    save_figure(fig, "task50_4_gray_hist")
    plt.close(fig)
    return {"hist": hist, "bin_edges": bin_edges}


# 5. Разбиение на 3 изображения по гистограмме
def split_by_histogram(gray, hist_info):
    hist, edges = hist_info["hist"], hist_info["bin_edges"]
    cum = np.cumsum(hist)
    total = cum[-1]
    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]
    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1:.1f})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1:.1f}–{t2:.1f})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2:.1f})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_5_split3")
    plt.close(fig)
    return (part1, part2, part3), (float(t1), float(t2))


# 6. Чёрная круговая рамка
def add_circular_frame(img):
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    R = min(h, w) / 2.0 - 2
    Y, X = np.ogrid[:h, :w]
    outside = (X - cx) ** 2 + (Y - cy) ** 2 > R ** 2
    framed = img.copy()
    framed[outside] = 0
    _before_after(img, framed, "В круговой рамке", "task50_6_frame")
    return framed


# 7. Случайный шум
def add_noise(img, sigma: float = 25.0):
    noisy = img.astype(np.float32) + np.random.normal(0.0, sigma, img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    _before_after(img, noisy, f"Шум (σ = {sigma})", "task50_7_noisy")
    return noisy


# 8. Гауссово размытие
def gaussian_blur(img, sigma: float = 2.0):
    blurred = np.stack([gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
                        for c in range(img.shape[2])], axis=-1)
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    _before_after(img, blurred, f"Размытие (σ = {sigma})", "task50_8_blurred")
    return blurred


# 9. Sharpening (unsharp masking)
def sharpen_image(img, sigma: float = 2.0, amount: float = 1.5):
    f = img.astype(np.float32)
    blurred = gaussian_filter(f, sigma=(sigma, sigma, 0))
    sharp = np.clip(f + amount * (f - blurred), 0, 255).astype(np.uint8)
    _before_after(img, sharp, f"Sharpening (amount={amount})", "task50_9_sharpened")
    return sharp


# 10. PCA
def apply_pca_to_image(img, patch: int = 8, n_components: int = 16):
    """10а: PCA в пространстве RGB (пиксель = вектор из 3 признаков).
    10б: PCA по блокам 8x8 серого изображения (вектор из 64 признаков):
    восстановление, кривая MSE и 16 главных компонент."""
    h, w = img.shape[:2]
    out = {}

    # --- 10а: RGB-пиксели ---
    flat = img.reshape(-1, 3).astype(np.float32)
    pca_rgb = PCA(n_components=3)
    restored = pca_rgb.inverse_transform(pca_rgb.fit_transform(flat))
    restored_img = np.clip(restored.reshape(h, w, 3), 0, 255).astype(np.uint8)
    _before_after(img, restored_img, "PCA в RGB (3 компоненты)", "task50_10a_pca_rgb")
    out["rgb_explained"] = pca_rgb.explained_variance_ratio_.astype(float)

    # --- 10б: блоки 8x8 ---
    gray = rgb2gray(img).astype(np.float32) * 255
    H, W = (h // patch) * patch, (w // patch) * patch
    g = gray[:H, :W]
    blocks = (g.reshape(H // patch, patch, W // patch, patch)
               .swapaxes(1, 2).reshape(-1, patch * patch))
    pca = PCA(n_components=n_components)
    rec_blocks = pca.inverse_transform(pca.fit_transform(blocks))
    rec = (rec_blocks.reshape(H // patch, W // patch, patch, patch)
                     .swapaxes(1, 2).reshape(H, W))
    rec = np.clip(rec, 0, 255)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(g, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Исходное (ч/б)")
    axes[1].imshow(rec, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"PCA по блокам {patch}×{patch}, {n_components} компонент")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_10b_pca_restored")
    plt.close(fig)

    # кривая MSE
    max_k = patch * patch
    full = PCA(n_components=max_k).fit(blocks)
    mse = []
    mean = full.mean_
    centered = blocks - mean
    proj = centered @ full.components_.T
    for k in range(1, max_k + 1):
        r = proj[:, :k] @ full.components_[:k] + mean
        mse.append(float(np.mean((blocks - r) ** 2)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), mse, marker="o", markersize=3)
    ax.set_xlabel("Число главных компонент")
    ax.set_ylabel("MSE восстановления")
    ax.set_title("Зависимость ошибки восстановления от числа компонент")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "task50_10c_pca_mse")
    plt.close(fig)

    # первые 16 компонент
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(full.components_[i].reshape(patch, patch), cmap="gray")
        ax.set_title(f"PC {i + 1}", fontsize=8)
        ax.axis("off")
    save_figure(fig, "task50_10d_pca_components")
    plt.close(fig)

    out.update(mse_curve=mse, mse_at=dict(k1=mse[0], k16=mse[15], k64=mse[-1]),
               block_explained=full.explained_variance_ratio_)
    return out


def run_task50(image_path: str) -> dict:
    img = plt.imread(image_path)
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = img[..., :3]
    print(f"[50] Изображение {image_path}: shape={img.shape}, dtype={img.dtype}")
    r = {}
    r["channels"] = split_channels(img);                print("     1. каналы R, G, B разделены")
    r["rotated"] = rotate_image(img, 40.0);              print(f"     2. поворот на 40°, новый размер {r['rotated'].shape}")
    r["gray"] = to_grayscale(img);                       print(f"     3. ч/б: shape={r['gray'].shape}, min={r['gray'].min()}, max={r['gray'].max()}")
    r["hist"] = gray_histogram(r["gray"]);               print("     4. гистограмма (256 бинов) построена")
    r["parts"], th = split_by_histogram(r["gray"], r["hist"])
    print(f"     5. разбиение по гистограмме: пороги t1={th[0]:.1f}, t2={th[1]:.1f}")
    r["framed"] = add_circular_frame(img);               print("     6. круговая рамка наложена")
    r["noisy"] = add_noise(img, 25.0);                   print("     7. гауссов шум, σ=25")
    r["blurred"] = gaussian_blur(img, 2.0);              print("     8. размытие, σ=2")
    r["sharpened"] = sharpen_image(img, 2.0, 1.5);       print("     9. sharpening, σ=2, amount=1.5")
    r["pca"] = apply_pca_to_image(img)
    e = r["pca"]
    print("     10. PCA: RGB-доли дисперсии =", np.round(e["rgb_explained"], 4).tolist())
    print(f"         блоки 8x8: MSE(k=1)={e['mse_at']['k1']:.2f}, MSE(k=16)={e['mse_at']['k16']:.2f}, MSE(k=64)={e['mse_at']['k64']:.2e}")
    print("Задание 50 выполнено. Все изображения сохранены в output/.")
    return r
