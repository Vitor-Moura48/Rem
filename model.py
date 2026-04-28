from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch
from torchvision import models
import matplotlib.pyplot as plt

import evaluation

# --- Transformações ---
def apply_transforms():

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),        # Redimensiona de acordo com o modelo pré-treinado
        transforms.ToTensor(),                # [0,255] HxWxC → [0,1] CxHxW (converte para formato PyTorch)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Média pré estabelecida para o modelo pré-treinado
                            std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms

def create_train_loader():
    # Conta imagens por classe
    class_counts = torch.zeros(len(train_dataset.classes))
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    # Pesos inversos à frequência
    class_weights = 1.0 / class_counts

    # Sampler — balanceia os batches
    sample_weights = torch.tensor([class_weights[label] for _, label in train_dataset.samples])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler

# --- Modelo ---
def build_model(num_classes):

    # Carrega VGG16 com pesos pré-treinados
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Adiciona uma tag nas camadas convolucionais (para indicar que não devem ser alteradas durante o treinamento)
    for param in model.features.parameters():
        param.requires_grad = False

    # Substitui a última camada do classificador para se adequar ao número de classes do problema
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(4096, num_classes)
    )

    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()        # Zera os gradientes acumulados
        outputs = model(images)      # Passa as imagens pelo modelo para obter as previsões
        loss = criterion(outputs, labels)  # calcula o erro
        loss.backward()              # backpropagation
        optimizer.step()             # atualiza os pesos

        total_loss += loss.item()
        preds = outputs.argmax(dim=1) 
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Acurácia
    ax1.plot(epochs, history['train_acc'], label='Treino')
    ax1.plot(epochs, history['test_acc'],  label='Teste')
    ax1.set_title('Acurácia')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Acurácia')
    ax1.legend()

    # Loss
    ax2.plot(epochs, history['train_loss'], label='Treino')
    ax2.plot(epochs, history['test_loss'],  label='Teste')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def apply_epochs(model: models.VGG, train_loader: DataLoader, test_loader: DataLoader, epochs: int):
    # Loss — penaliza mais os erros nas minoritárias
    #class_weights_normalized = (class_weights / class_weights.sum()).to(device)
    criterion = nn.CrossEntropyLoss() # weight=class_weights_normalized
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(epochs):

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.3f}")
    
    torch.save(model.state_dict(), 'sleep_classifier.pth')
    
    return history


if __name__ == "__main__":

    train_transforms, test_transforms = apply_transforms()

    # --- Dataset: lê as pastas automaticamente ---
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transforms)
    test_dataset  = datasets.ImageFolder(root='dataset/test',  transform=test_transforms)

    # --- DataLoader: cria os batches ---
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    # Constroi a estrutura do modelo
    NUM_CLASSES = len(train_dataset.classes)
    model = build_model(num_classes=NUM_CLASSES)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ------------- apenas um desses blocos deve estar habilitado -------------
    # train_loader = create_train_loader()
    # history = apply_epochs(model, train_loader, test_loader, 20)
    # plot_training_history(history)

    model.load_state_dict(torch.load('sleep_classifier.pth', map_location=device))
    # -------------                                               -------------

    # Avaliação completa (métricas por classe + matriz de confusão)
    metrics = evaluation.full_evaluation(model, test_loader, test_dataset, device)