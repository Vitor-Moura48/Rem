import torch
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm


class CachedImageFolder(Dataset):

    def __init__(self, image_folder: datasets.ImageFolder):
        self.classes = image_folder.classes
        self.samples = image_folder.samples
        self.class_to_idx = image_folder.class_to_idx

        # Pré-carrega tudo na RAM
        self.cached_images = []
        self.cached_labels = []

        print(f"[Cache] Pré-carregando {len(image_folder)} imagens na RAM...")
        for idx in tqdm(range(len(image_folder)), desc="Caching", unit="img"):
            image, label = image_folder[idx]  # Já aplica o transform
            
            # Converte para uint8 (0 a 255) reduz o consumo de RAM (4 bytes -> 1 byte por pixel)
            image_uint8 = (image * 255).to(torch.uint8)
            self.cached_images.append(image_uint8)
            self.cached_labels.append(label)

        # Empilha todos os labels num tensor único para acesso rápido
        self.cached_labels = torch.tensor(self.cached_labels, dtype=torch.long)
        
        # Move tudo para Memória Dedicada da GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cached_images = torch.stack(self.cached_images).to(device)
        self.cached_labels = self.cached_labels.to(device)

        print(f"[Cache] Pronto! {len(self.cached_images)} imagens em memória VRAM da GPU.")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        # Descomprime o tensor de volta para float32 (0 a 1) na hora de treinar
        img_float = self.cached_images[idx].to(torch.float32) / 255.0
        return img_float, self.cached_labels[idx]


class CachedConcatDataset(Dataset):

    def __init__(self, image_folders: list, augmentations=None, max_oversample=3, balance=True):
        self.augmentations = augmentations

        # Combina metadados de todos os datasets
        self.classes = image_folders[0].classes
        self.class_to_idx = image_folders[0].class_to_idx
        self.samples = []
        for ds in image_folders:
            self.samples.extend(ds.samples)

        # Pré-carrega tudo na RAM
        self.cached_images = []
        self.cached_labels = []

        total = sum(len(ds) for ds in image_folders)
        print(f"[Cache] Pré-carregando {total} imagens de {len(image_folders)} datasets na RAM...")

        with tqdm(total=total, desc="Caching", unit="img") as pbar:
            for ds in image_folders:
                for idx in range(len(ds)):
                    image, label = ds[idx]  # Já aplica o transform
                    
                    # Comprime para uint8
                    image_uint8 = (image * 255).to(torch.uint8)
                    self.cached_images.append(image_uint8)
                    self.cached_labels.append(label)
                    pbar.update(1)

        self.cached_labels = torch.tensor(self.cached_labels, dtype=torch.long)

        # Balanceia as classes (treino) ou deixa original (teste)
        if balance:
            self._balance_classes(max_oversample)
            
        # Move tudo para Memória Dedicada da GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cached_images = torch.stack(self.cached_images).to(device)
        self.cached_labels = self.cached_labels.to(device)

        if balance:
            print(f"[Cache] Pronto! {len(self.cached_images)} imagens em memória VRAM (balanceadas).")
        else:
            print(f"[Cache] Pronto! {len(self.cached_images)} imagens de teste em memória VRAM (originais).")

    def _balance_classes(self, max_oversample):
        import random

        num_classes = len(self.classes)
        
        # Agrupa os índices por classe
        indices_per_class = {c: [] for c in range(num_classes)}
        for i, label in enumerate(self.cached_labels.tolist()):
            indices_per_class[label].append(i)
        
        # Encontra a menor classe e calcula o teto
        min_count = min(len(idxs) for idxs in indices_per_class.values())
        target = min_count * max_oversample
        
        print(f"[Balanceamento] Menor classe: {min_count} imgs | Teto (x{max_oversample}): {target} imgs")
        for c in range(num_classes):
            count = len(indices_per_class[c])
            if count < target:
                print(f"  Classe {self.classes[c]:>5}: {count:>5} -> {target} (+{target - count} duplicações)")
            elif count > target:
                print(f"  Classe {self.classes[c]:>5}: {count:>5} -> {target} (-{count - target} descartadas)")
            else:
                print(f"  Classe {self.classes[c]:>5}: {count:>5} -> {target} (sem alteração)")
        
        # Monta a lista final de índices balanceados
        balanced_indices = []
        for c in range(num_classes):
            idxs = indices_per_class[c]
            if len(idxs) <= target:
                
                # Classe menor ou igual ao teto: pega tudo + duplica aleatoriamente até o teto
                balanced_indices.extend(idxs)
                extras_needed = target - len(idxs)
                if extras_needed > 0:
                    balanced_indices.extend(random.choices(idxs, k=extras_needed))
            else:
                # Classe maior que o teto: sorteia aleatoriamente sem repetição
                balanced_indices.extend(random.sample(idxs, k=target))
        
        # Reconstrói os caches com a ordem balanceada
        self.cached_images = [self.cached_images[i] for i in balanced_indices]
        self.cached_labels = self.cached_labels[balanced_indices]
        
        # Atualiza o samples para manter compatibilidade
        original_samples = list(self.samples)
        self.samples = [original_samples[i] for i in balanced_indices]

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        # Descomprime de volta para float32 no momento de usar
        img = self.cached_images[idx].to(torch.float32) / 255.0
        
        # Aplica as perturbações dinâmicas de forma aleatória e extremamente rápida
        if self.augmentations is not None:
            img = self.augmentations(img)
            
        return img, self.cached_labels[idx]

