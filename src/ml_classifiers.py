import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

class TorchKNN:
    def __init__(self, n_neighbors=5, device=None):
        self.k = n_neighbors
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X_train, y_train):
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        
    @torch.no_grad()
    def predict(self, X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        preds = []
        batch_size = 256
        for i in range(0, X_test.size(0), batch_size):
            x_batch = X_test[i:i+batch_size]
            dist = torch.cdist(x_batch, self.X_train)
            topk = dist.topk(self.k, largest=False)
            nearest_labels = self.y_train[topk.indices]
            batch_preds, _ = torch.mode(nearest_labels, dim=1)
            preds.extend(batch_preds.cpu().numpy())
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()

from sklearn.preprocessing import StandardScaler

class MLPipeline:
    def __init__(self, X_train, y_train, X_test, y_test, class_names):
        # Escalonamento dos dados (Crucial para Redes Neurais e KNN)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        
        # Modelos clássicos de Machine Learning habilitados
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', n_jobs=-1, random_state=42),
            "KNN_GPU": TorchKNN(n_neighbors=5),
            "NeuralNet": MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam', max_iter=300, early_stopping=False, random_state=42)
        }


    def train_and_evaluate(self, fold_name, save_dir='metrics'):
        for name, clf in self.models.items():
            model_dir = f"{save_dir}/{name}_{fold_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"  [{name}] Treinando...")
            clf.fit(self.X_train, self.y_train)
            preds = clf.predict(self.X_test)
            
            train_acc = clf.score(self.X_train, self.y_train)
            test_acc = clf.score(self.X_test, self.y_test)
            print(f"  [{name}] Acc Treino: {train_acc:.3f} | Acc Teste: {test_acc:.3f}")
            
            self._save_metrics(preds, model_dir, name, train_acc, test_acc)

    def _save_metrics(self, preds, out_dir, model_name, train_acc, test_acc):
        # Relatório de texto
        report = classification_report(self.y_test, preds, target_names=self.class_names)
        with open(f"{out_dir}/report.txt", "w") as f:
            f.write(f"Modelo: {model_name}\nAcc Treino: {train_acc:.4f} | Acc Teste: {test_acc:.4f}\n\n{report}")
            
        # Matriz de Confusão Absoluta
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f"{model_name} - Matriz de Confusão")
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cm.png")
        
        # Matriz de Confusão Normalizada
        cm_norm = cm.astype('float') / cm.sum(axis=1).reshape(-1, 1)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f"{model_name} - Matriz Normalizada")
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cm_normalized.png")
        
        plt.close('all')
