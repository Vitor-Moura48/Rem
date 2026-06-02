import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from model_factory import ModelFactory

class FeatureExtractor:
    def __init__(self, model_name: str, model_path: str, num_classes: int, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carrega o modelo treinado
        self.model = ModelFactory.build_model(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Remove apenas a última camada de classificação para cuspir as 84 features densas
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _extract(self, loader, desc=""):
        features, labels = [], []

        for images, targets in tqdm(loader, desc=desc, unit="batch", leave=False):
            out = self.model(images.to(self.device))

            if out.dim() > 2:
                out = out.view(out.size(0), -1)

            features.append(out.cpu().numpy())
            labels.append(targets.cpu().numpy())
            
        return np.concatenate(features), np.concatenate(labels)

    def extract_and_save(self, train_loader, test_loader, paths, fold_name):
        X_train, y_train = self._extract(train_loader, desc=f"Extraindo Treino {fold_name}")
        X_test, y_test = self._extract(test_loader, desc=f"Extraindo Teste {fold_name}")
        
        # Salva no disco
        np.save(paths[0], X_train)
        np.save(paths[1], y_train)
        np.save(paths[2], X_test)
        np.save(paths[3], y_test)
        
        return X_train, y_train, X_test, y_test
