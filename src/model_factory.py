import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

MODEL_CONFIGS = {
    "vgg16": {
        "input_size": (224, 224),
        "grayscale": True,
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },
    "lenet": {
        "input_size": (224, 224),
        "grayscale": True,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
}


class LeNet(nn.Module):

    def __init__(self, num_classes, input_size=(90, 551)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),  nn.BatchNorm2d(6),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),  nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size[0], input_size[1])
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_size, 120), nn.ReLU(),
            nn.Linear(120, 84),        nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ModelFactory:
    
    @staticmethod
    def get_config(model_name):
        return MODEL_CONFIGS.get(model_name, None)
    
    @staticmethod
    def get_transforms(model_name: str):

        config = MODEL_CONFIGS.get(model_name)
        if config is None:
            raise ValueError(f"Modelo desconhecido: {model_name}")

        pipeline = []

        if config["grayscale"]:
            pipeline.append(transforms.Grayscale(num_output_channels=1))

        # Redimensiona e converte para tensor (0 a 1 float32)
        pipeline.extend([
            transforms.Resize(config["input_size"]),
            transforms.ToTensor(),
        ])

        composed = transforms.Compose(pipeline)
        return composed, composed

    @staticmethod
    def get_augmentations():

        return transforms.Compose([

            # SpecAugment: Apaga aleatoriamente até 2 blocos do espectrograma (tempo ou frequência)
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            
            # Ruído Gaussiano
            AddGaussianNoise(mean=0., std=0.1)
        ])

    @staticmethod
    def build_vgg16(num_classes):

        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        for param in model.features.parameters():
            param.requires_grad = False
        
        # Descongela o último bloco conv (features 24 em diante)
        for param in model.features[24:].parameters():
            param.requires_grad = True

        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(4096, num_classes)
        )

        return model
    
    @staticmethod
    def build_lenet(num_classes):
        config = ModelFactory.get_config("lenet")
        input_size = config.get("input_size", (90, 621))
        model = LeNet(num_classes, input_size=input_size)
        return model


    BUILDERS = {
        "vgg16": build_vgg16,
        "lenet": build_lenet,
    }

    @staticmethod
    def build_model(model_name: str, num_classes: int):
        builder = ModelFactory.BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(f"Modelo desconhecido: {model_name}")
        return builder(num_classes)
