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
            self.cached_images.append(image)
            self.cached_labels.append(label)

        # Empilha todos os labels num tensor único para acesso rápido
        self.cached_labels = torch.tensor(self.cached_labels, dtype=torch.long)
        print(f"[Cache] Pronto! {len(self.cached_images)} imagens em memória.")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        return self.cached_images[idx], self.cached_labels[idx]


class CachedConcatDataset(Dataset):

    def __init__(self, image_folders: list):
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
                    self.cached_images.append(image)
                    self.cached_labels.append(label)
                    pbar.update(1)

        self.cached_labels = torch.tensor(self.cached_labels, dtype=torch.long)
        print(f"[Cache] Pronto! {len(self.cached_images)} imagens em memória.")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        return self.cached_images[idx], self.cached_labels[idx]
