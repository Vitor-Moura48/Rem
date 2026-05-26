import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model_factory import ModelFactory

class SleepStageClassifier():

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, model_name: str, device: torch.device = None):
        
        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # Obtém configuração do modelo (batch_size, lr, etc.)
        config = ModelFactory.get_config(model_name)
        
        self.batch_size = config["batch_size"]
        lr = config["lr"]

        # --- DataLoader: cria os batches ---
        self.sampler = self.create_custom_sampler()
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False, pin_memory=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, pin_memory=True)

        # Constroi a estrutura do modelo e move para o dispositivo
        num_classes = len(self.train_dataset.classes)
        self.model = ModelFactory.build_model(model_name=model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Configura o otimizador (apenas para as camadas que serão treinadas)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        # Configura a função de perda
        self.criterion = nn.CrossEntropyLoss()

    def create_custom_sampler(self):
        # Conta imagens por classe
        class_counts = torch.zeros(len(self.train_dataset.classes))
        for _, label in self.train_dataset.samples:
            class_counts[label] += 1

        # Pesos inversos à frequência
        class_weights = 1.0 / class_counts

        # Sampler — balanceia os batches
        sample_weights = torch.tensor([class_weights[label] for _, label in self.train_dataset.samples])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return sampler

    def apply_epochs(self, epochs=20, directory='models', name='vgg16_finetuned.pth', save_history=True, history_dir='metrics'):

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        for epoch in range(epochs):

            train_loss, train_acc = self.train_epoch()
            test_loss,  test_acc  = self.evaluate_epoch()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.3f}")
        
        os.makedirs(directory, exist_ok=True)
        
        torch.save(self.model.state_dict(), f'{directory}/{name}')
        
        if save_history:
            self.save_training_history(history, filename=name.replace('.pth', '.png'), save_dir=history_dir)


    def train_epoch(self):

        self.model.train()
        total_loss, correct = 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()              # Zera os gradientes acumulados
            outputs = self.model(images)            # Passa as imagens pelo modelo para obter as previsões
            loss = self.criterion(outputs, labels)  # calcula o erro
            loss.backward()                         # backpropagation
            self.optimizer.step()                   # atualiza os pesos

            total_loss += loss.item()
            preds = outputs.argmax(dim=1) 
            correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / len(self.train_loader.dataset)
        return avg_loss, accuracy

    def evaluate_epoch(self):
        self.model.eval()
        total_loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        return total_loss / len(self.test_loader), correct / len(self.test_loader.dataset)

    def save_training_history(self, history, filename, save_dir='metrics'):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

        # Acurácia
        ax1.plot(epochs, history['train_acc'], label='Treino')
        ax1.plot(epochs, history['test_acc'],  label='Teste')
        ax1.set_title('Acurácia')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Acurácia')
        ax1.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/{filename}')
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def evaluate_model(self, save_dir='metrics'):
    
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        class_names = self.test_dataset.classes
        
        # --- Dados brutos ---
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print(report)
        
        # Salva o report em txt
        with open(f'{save_dir}/classification_report.txt', 'w') as f:
            f.write(report)
        
        # Métricas por classe em dict
        metrics = {
            'precision': precision_score(all_labels, all_preds, average=None),
            'recall':    recall_score(all_labels, all_preds, average=None),
            'f1':        f1_score(all_labels, all_preds, average=None),
        }
        
        # --- Matriz de confusão ---
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png')

        # Matriz de confusão normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1).reshape(-1, 1)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_normalized.png')
        
        # --- Gráfico de métricas por classe ---
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')

        data = []
        for i, class_name in enumerate(class_names):
            data.append([
                class_name,
                f"{metrics['precision'][i]:.3f}",
                f"{metrics['recall'][i]:.3f}",
                f"{metrics['f1'][i]:.3f}"
            ])

        table = ax.table(
            cellText=data,
            colLabels=['Classe', 'Precisão', 'Recall', 'F1'],
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        plt.title('Métricas por Classe', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_table.png')