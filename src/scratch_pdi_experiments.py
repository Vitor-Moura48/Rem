import os
import gc
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader
from PIL import Image
from skimage import exposure
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

from model_factory import ModelFactory
from model import SleepStageClassifier
from cached_dataset import CachedImageFolder, CachedConcatDataset

# ---------------------------------------------------------
# Kernel Gaussiano compartilhado
# ---------------------------------------------------------
GAUSSIAN_KERNEL = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
GAUSSIAN_KERNEL /= GAUSSIAN_KERNEL.sum()

# ---------------------------------------------------------
# Transforms Customizados (operam em grayscale, 1 canal)
# ---------------------------------------------------------
class HistEqTransform:
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        out = exposure.equalize_hist(img_np)
        return Image.fromarray((out * 255).astype(np.uint8)).convert('L')

class HighBoostTransform:
    def __init__(self, amount=0.8):
        self.amount = amount
    def __call__(self, img):
        img_np = np.array(img.convert('L')).astype(np.float32)
        blurred = convolve2d(img_np, GAUSSIAN_KERNEL, mode='same', boundary='symm')
        details = img_np - blurred
        out = img_np + self.amount * details
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert('L')

class HistEqAndHighBoostTransform:
    def __init__(self, amount=0.8):
        self.amount = amount
    def __call__(self, img):
        img_np = np.array(img.convert('L'))
        eq = (exposure.equalize_hist(img_np) * 255).astype(np.float32)
        blurred = convolve2d(eq, GAUSSIAN_KERNEL, mode='same', boundary='symm')
        details = eq - blurred
        out = eq + self.amount * details
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert('L')


# ---------------------------------------------------------
# Função para Salvar Exemplos (Debug)
# ---------------------------------------------------------
def save_debug_images(dataset_dir, subject):
    print("\n[Debug] Salvando imagens de exemplo das transformações...")
    os.makedirs('debug_filters', exist_ok=True)
    
    classes = ['Sleep_stage_W', 'Sleep_stage_N1', 'Sleep_stage_N2', 'Sleep_stage_N3', 'Sleep_stage_R']
    transforms_dict = {
        'original': lambda x: x.convert('L'),
        'histeq': HistEqTransform(),
        'highboost': HighBoostTransform(),
        'both': HistEqAndHighBoostTransform()
    }
    
    for cls in classes:
        cls_dir = f"{dataset_dir}/{subject}/{cls}"
        if not os.path.exists(cls_dir): continue
        images = os.listdir(cls_dir)
        if len(images) == 0: continue
        
        img_path = f"{cls_dir}/{images[0]}"
        img = Image.open(img_path)
        short_name = cls.replace('Sleep_stage_', '')
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, (name, tfm) in enumerate(transforms_dict.items()):
            res = tfm(img)
            axes[i].imshow(res, cmap='gray')
            axes[i].set_title(f"{short_name} - {name}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"debug_filters/sample_{short_name}.png")
        plt.close()
    print("Exemplos salvos na pasta 'debug_filters/'.\n")

# ---------------------------------------------------------
# Fluxo Principal
# ---------------------------------------------------------
if __name__ == "__main__":
    dataset_dir = 'spectrograms'
    backbone = 'lenet'
    
    subjects = sorted(os.listdir(dataset_dir))
    dummy_ds = datasets.ImageFolder(root=f'{dataset_dir}/{subjects[0]}')
    class_names = dummy_ds.classes
    num_classes = len(class_names)
    
    # 1. Salvar amostras de debug usando o primeiro sujeito
    save_debug_images(dataset_dir, subjects[0])
    
    # K-Fold logic (Focaremos apenas no FOLD 1 para agilidade)
    subjects_per_fold = 8
    test_group = subjects[0:subjects_per_fold]
    train_subjects = [s for s in subjects if s not in test_group]
    
    print(f"==================================================")
    print(f" INICIANDO EXPERIMENTOS PDI NO FOLD 01")
    print(f" Teste: {test_group}")
    print(f"==================================================")

    configs = {
        'histeq': HistEqTransform(),
        'highboost': HighBoostTransform(),        
        'histeq_highboost': HistEqAndHighBoostTransform() 
    }
    
    base_train_tfm, base_test_tfm = ModelFactory.get_transforms(backbone)
    augmentations = ModelFactory.get_augmentations()
    
    for config_name, custom_tfm in configs.items():
        print(f"\n[{config_name.upper()}] Montando e cacheando datasets na RAM (aguarde)...")
        
        # Injeta o filtro customizado ANTES das transforms originais da LeNet (Grayscale, Resize, ToTensor)
        train_pipeline = transforms.Compose([
            custom_tfm,
            *base_train_tfm.transforms,
        ])
        test_pipeline = transforms.Compose([
            custom_tfm,
            *base_test_tfm.transforms
        ])
        
        raw_train = [
            datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=train_pipeline) for s in train_subjects
        ]
        train_dataset = CachedConcatDataset(
            raw_train,
            augmentations=ModelFactory.get_augmentations(),
            max_oversample=1
        )
        
        if len(test_group) == 1:
            raw_test = datasets.ImageFolder(root=f'{dataset_dir}/{test_group[0]}', transform=test_pipeline)
            test_dataset = CachedImageFolder(raw_test)
        else:
            raw_test = [
                datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_pipeline) for s in test_group
            ]
            test_dataset = CachedConcatDataset(raw_test, augmentations=None, balance=False)
        
        print(f"[{config_name.upper()}] Cache concluído. Inicializando a LeNet...")
        classifier = SleepStageClassifier(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_name=backbone
        )
        
        metrics_dir = f"metrics/pdi_{config_name}"
        model_save_path = f"models/pdi_{config_name}_fold01.pth"
        
        print(f"[{config_name.upper()}] Iniciando Treinamento (20 épocas padrão)...")
        # Treina
        classifier.apply_epochs(
            epochs=20, 
            directory='models',
            name=f'pdi_{config_name}_fold01.pth',
            save_history=True,
            history_dir=metrics_dir
        )
        
        # Avalia (Gera as matrizes de confusão e relatórios na pasta metrics/pdi_xxx/)
        print(f"[{config_name.upper()}] Extraindo as Métricas Finais...")
        classifier.load_model(model_save_path)
        classifier.evaluate_model(save_dir=metrics_dir)
        
        del classifier, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\n==================================================")
    print("TODOS OS EXPERIMENTOS PDI FORAM CONCLUÍDOS!")
    print("Você encontrará as tabelas e gráficos separados nas pastas:")
    print(" - metrics/pdi_histeq/")
    print(" - metrics/pdi_highboost/")
    print(" - metrics/pdi_histeq_highboost/")
    print("==================================================")
