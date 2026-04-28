import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

def full_evaluation(model, loader, dataset, device, save_dir='metrics'):
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = dataset.classes
    
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
    plt.show()
    
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
    plt.show()
        
    return metrics