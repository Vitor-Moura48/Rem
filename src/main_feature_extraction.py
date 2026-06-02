import os
import gc
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset

from model_factory import ModelFactory
from feature_extractor import FeatureExtractor
from ml_classifiers import MLPipeline

if __name__ == "__main__":
    backbone = "lenet"
    dataset_dir = 'spectrograms'
    features_dir = 'features'
    models_dir = 'models'
    
    os.makedirs(features_dir, exist_ok=True)
    subjects = sorted(os.listdir(dataset_dir))
    
    # Extrair classes do primeiro sujeito disponível
    dummy_ds = datasets.ImageFolder(root=f'{dataset_dir}/{subjects[0]}')
    class_names = dummy_ds.classes
    num_classes = len(class_names)
    
    train_transforms, test_transforms = ModelFactory.get_transforms(backbone)
    
    # K-Fold mirroring main.py
    subjects_per_fold = 8
    groups = [subjects[i : i + subjects_per_fold] for i in range(0, len(subjects), subjects_per_fold)]
    
    for fold_idx, test_group in enumerate(groups):
        fold_id = fold_idx + 1
        fold_name = f"fold{fold_id:02d}"
        print(f"\n{'='*50}\n  FOLD {fold_id:02d} (Teste: {test_group})\n{'='*50}")
        
        # Arquivos a serem salvos
        feat_paths = [f"{features_dir}/{fold_name}_{s}.npy" for s in ["X_train", "y_train", "X_test", "y_test"]]
        
        # Pipeline de Extração (Verifica se já existe no disco)
        if all(os.path.exists(p) for p in feat_paths):
            print("[Info] Carregando features pré-computadas do disco...")
            X_train, y_train, X_test, y_test = [np.load(p) for p in feat_paths]

        else:
            model_pth = f'{models_dir}/{backbone}_finetuned{fold_id:02d}.pth'
            if not os.path.exists(model_pth):
                print(f"[Aviso] Pesos '{model_pth}' não encontrados. Execute o main.py primeiro. Pulando...")
                continue
                
            print("[Info] Extraindo features da LeNet (serão salvas no disco)...")
            train_subjects = [s for s in subjects if s not in test_group]
            
            # Usamos test_transforms (sem data augmentation) para extração de features
            train_dataset = ConcatDataset([
                datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_transforms) for s in train_subjects
            ])
            
            if len(test_group) == 1:
                test_dataset = datasets.ImageFolder(root=f'{dataset_dir}/{test_group[0]}', transform=test_transforms)
            else:
                test_dataset = ConcatDataset([
                    datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_transforms) for s in test_group
                ])
            
            # Loaders apenas para iterar rapidamente sem shuffle
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=False)
            
            # Extrai e salva
            extractor = FeatureExtractor(backbone, model_pth, num_classes)
            X_train, y_train, X_test, y_test = extractor.extract_and_save(train_loader, test_loader, feat_paths, fold_name)
            
            del train_dataset, test_dataset, train_loader, test_loader, extractor
            gc.collect()
            torch.cuda.empty_cache()
            
        # Pipeline de Machine Learning
        pipeline = MLPipeline(X_train, y_train, X_test, y_test, class_names)
        pipeline.train_and_evaluate(fold_name=fold_name)
        
        del pipeline, X_train, y_train, X_test, y_test
        gc.collect()
