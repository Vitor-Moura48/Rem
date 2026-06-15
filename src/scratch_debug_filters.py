"""
Script descartável - apenas gera imagens de debug dos filtros PDI.
Não treina nada, apenas salva exemplos visuais de cada classe.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
from scipy.signal import convolve2d

GAUSSIAN_KERNEL = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
GAUSSIAN_KERNEL /= GAUSSIAN_KERNEL.sum()

dataset_dir = 'spectrograms'
output_dir  = 'debug_filters'
os.makedirs(output_dir, exist_ok=True)

class_names = ['Sleep_stage_N1', 'Sleep_stage_N2', 'Sleep_stage_N3', 'Sleep_stage_R', 'Sleep_stage_W']

def apply_histeq(img):
    img_np = np.array(img.convert('L'))  # 1 canal (grayscale)
    out = exposure.equalize_hist(img_np)
    return Image.fromarray((out * 255).astype(np.uint8)).convert('L')

def apply_highboost(img, amount=2.0):
    img_np = np.array(img.convert('L')).astype(np.float32)
    blurred = convolve2d(img_np, GAUSSIAN_KERNEL, mode='same', boundary='symm')
    details = img_np - blurred
    out = img_np + amount * details
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert('L')

def apply_both(img, amount=2.0):
    # 1. HistEq
    img_np = np.array(img.convert('L'))
    eq = (exposure.equalize_hist(img_np) * 255).astype(np.float32)
    # 2. High Boost
    blurred = convolve2d(eq, GAUSSIAN_KERNEL, mode='same', boundary='symm')
    details = eq - blurred
    out = eq + amount * details
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert('L')

subjects = sorted(os.listdir(dataset_dir))

for cls in class_names:
    img_path = None
    # Procura a primeira imagem disponível dessa classe em qualquer sujeito
    for subj in subjects:
        cls_dir = os.path.join(dataset_dir, subj, cls)
        if os.path.isdir(cls_dir):
            files = os.listdir(cls_dir)
            if files:
                img_path = os.path.join(cls_dir, files[0])
                break

    if img_path is None:
        print(f"[!] Nenhuma imagem encontrada para {cls}")
        continue

    img = Image.open(img_path)
    short_name = cls.replace('Sleep_stage_', '')

    original  = img.convert('L')
    histeq    = apply_histeq(img)
    highboost = apply_highboost(img)
    combined  = apply_both(img)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Classe: {short_name}', fontsize=14, fontweight='bold')

    for ax, image, title in zip(axes,
                                [original, histeq, highboost, combined],
                                ['Original', 'Equalização de Histograma', 'High-Boost', 'HistEq + High-Boost']):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{short_name}_filters_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {short_name} -> {save_path}")

print(f"\nPronto! Todas as imagens de debug estão em: '{output_dir}/'")
