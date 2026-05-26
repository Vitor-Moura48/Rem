from torchvision import transforms
from torchvision import models
import torch.nn as nn

MODEL_CONFIGS = {
    "vgg16": {
        "input_size": (224, 224),
        "grayscale": False,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "batch_size": 32,
        "lr": 1e-4,
    },
    "resnet18": {
        "input_size": (224, 224),
        "grayscale": False,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "batch_size": 32,
        "lr": 1e-4,
    },
    "efficientnet_b0": {
        "input_size": (224, 224),
        "grayscale": False,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "batch_size": 32,
        "lr": 1e-4,
    },
    "lenet": {
        "input_size": (32, 32),
        "grayscale": True,
        "normalize_mean": [0.5],
        "normalize_std": [0.5],
        "batch_size": 64,
        "lr": 1e-3,
    },
}


class LeNet(nn.Module):
    """LeNet-5 adaptada para classificação de espectrogramas (1 canal, 32x32)."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),  nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84),         nn.ReLU(),
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

        pipeline.extend([
            transforms.Resize(config["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["normalize_mean"], std=config["normalize_std"])
        ])

        composed = transforms.Compose(pipeline)
        return composed, composed

    @staticmethod
    def build_vgg16(num_classes):

        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(4096, num_classes)
        )

        return model
    
    @staticmethod
    def build_resnet18(num_classes):
        
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in list(model.parameters())[:-2]:  # Congela tudo exceto a última camada
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    @staticmethod
    def build_efficientnet_b0(num_classes):

        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        return model

    @staticmethod
    def build_lenet(num_classes):

        model = LeNet(num_classes)
        return model


    BUILDERS = {
        "vgg16": build_vgg16,
        "resnet18": build_resnet18,
        "efficientnet_b0": build_efficientnet_b0,
        "lenet": build_lenet,
    }

    @staticmethod
    def build_model(model_name: str, num_classes: int):
        builder = ModelFactory.BUILDERS.get(model_name)
        if builder is None:
            raise ValueError(f"Modelo desconhecido: {model_name}")
        return builder(num_classes)
