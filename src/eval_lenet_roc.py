import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

from model_factory import ModelFactory

if __name__ == "__main__":
    dataset_dir = 'spectrograms'
    models_dir = 'models'
    backbone = 'lenet'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando: {device}")
    
    subjects = sorted(os.listdir(dataset_dir))
    dummy_ds = datasets.ImageFolder(root=f'{dataset_dir}/{subjects[0]}')
    class_names = dummy_ds.classes
    num_classes = len(class_names)
    
    _, test_transforms = ModelFactory.get_transforms(backbone)
    
    subjects_per_fold = 8
    groups = [subjects[i : i + subjects_per_fold] for i in range(0, len(subjects), subjects_per_fold)]
    
    all_tpr = []
    all_roc_auc = []
    
    # Estruturas para guardar as curvas de cada classe individualmente
    class_tpr = {i: [] for i in range(num_classes)}
    class_auc = {i: [] for i in range(num_classes)}
    
    # Array de taxas de falsos positivos padrão para podermos interpolar e fazer a média das curvas
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_idx, test_group in enumerate(groups):
        fold_id = fold_idx + 1
        fold_name = f"fold{fold_id:02d}"
        model_pth = f'{models_dir}/{backbone}_finetuned{fold_id:02d}.pth'
        
        if not os.path.exists(model_pth):
            print(f"Ignorando {fold_name}: pesos '{model_pth}' não encontrados.")
            continue
            
        print(f"\nAvaliando {fold_name}...")
        
        # Monta dataloader de teste para esse fold
        if len(test_group) == 1:
            test_dataset = datasets.ImageFolder(root=f'{dataset_dir}/{test_group[0]}', transform=test_transforms)
        else:
            test_dataset = ConcatDataset([
                datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_transforms) for s in test_group
            ])
            
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Instancia e carrega a LeNet Full
        model = ModelFactory.build_lenet(num_classes)
        model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        y_true = []
        y_scores = []
        
        # Inferência
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Transforma logit em probabilidade
                probs = F.softmax(outputs, dim=1)
                
                y_scores.extend(probs.cpu().numpy())
                y_true.extend(labels.numpy())
                
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Cálculo do AUC para o Fold (Classe a Classe)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            # One-vs-Rest (1 se for a classe i, 0 caso contrário)
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Interpola a curva desta classe para a média
            interp_tpr_class = np.interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr_class[0] = 0.0
            class_tpr[i].append(interp_tpr_class)
            class_auc[i].append(roc_auc[i])
            
        # Calcula Macro-average da Curva ROC do fold
        all_fpr_fold = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr_fold = np.zeros_like(all_fpr_fold)
        
        for i in range(num_classes):
            mean_tpr_fold += np.interp(all_fpr_fold, fpr[i], tpr[i])
            
        mean_tpr_fold /= num_classes
        
        macro_auc = auc(all_fpr_fold, mean_tpr_fold)
        print(f"  -> Macro AUC do {fold_name}: {macro_auc:.4f}")
        
        # Interpolação no grid global (mean_fpr) para poder fazer a média final entre todos os folds
        interp_tpr = np.interp(mean_fpr, all_fpr_fold, mean_tpr_fold)
        interp_tpr[0] = 0.0
        all_tpr.append(interp_tpr)
        all_roc_auc.append(macro_auc)
    
    if all_tpr:
        # ---------------------------------------------------------
        # PLOT 1: Média Macro entre Folds
        # ---------------------------------------------------------
        mean_tpr = np.mean(all_tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(all_roc_auc)
        
        print(f"\n{'='*40}")
        print(f"Macro AUC Final ({len(all_tpr)} folds): {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"{'='*40}")
        
        os.makedirs('metrics', exist_ok=True)
        plt.figure(figsize=(8, 6))
        
        for i, (tpr_fold, auc_fold) in enumerate(zip(all_tpr, all_roc_auc)):
            plt.plot(mean_fpr, tpr_fold, lw=1, alpha=0.4, label=f'ROC Fold {i+1} (AUC = {auc_fold:.3f})')
            
        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.9,
                 label=r'Média Macro ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc))
                 
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Aleatório (AUC = 0.5)')
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC - Média dos Folds (LeNet)')
        plt.legend(loc="lower right")
        
        save_path_macro = 'metrics/lenet_macro_roc_curve.png'
        plt.savefig(save_path_macro, dpi=150, bbox_inches='tight')
        plt.close()
        
        # ---------------------------------------------------------
        # PLOT 2: Média por Classe (A mais explicativa)
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i in range(num_classes):
            mean_tpr_class = np.mean(class_tpr[i], axis=0)
            mean_tpr_class[-1] = 1.0
            mean_auc_class = auc(mean_fpr, mean_tpr_class)
            std_auc_class = np.std(class_auc[i])
            
            plt.plot(mean_fpr, mean_tpr_class, color=colors[i], lw=2,
                     label=f'Classe {class_names[i]} (AUC = {mean_auc_class:.3f} $\pm$ {std_auc_class:.3f})')
                     
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC por Classe do Sono (Média dos Folds)')
        plt.legend(loc="lower right")
        
        save_path_classes = 'metrics/lenet_classes_roc_curve.png'
        plt.savefig(save_path_classes, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gráficos consolidados salvos na pasta 'metrics/'!")
    else:
        print("Nenhum modelo foi avaliado. Verifique a pasta 'models/'.")
