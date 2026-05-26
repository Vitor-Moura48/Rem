import os
import gc
import torch

from spectrogram_converter import SpectrogramConverter
from data_splitter import DataSplitter
from model import SleepStageClassifier
from torchvision import datasets
from cached_dataset import CachedImageFolder, CachedConcatDataset
from model_factory import ModelFactory

if __name__ == "__main__":
    """
    converter = SpectrogramConverter()

    paths = [
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4011E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4011EH-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4021E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4021EH-Hypnogram.edf"),
    ]

    for edf_path, hypnogram_path in paths:
        converter.convert(edf_path=edf_path, hypnogram_path=hypnogram_path)
    """


    # Divide os dados em treino e teste (Não utilizado ao aplicar o LOSO)
    #splitter = DataSplitter()
    #splitter.split(seed=42)


    model_name = "lenet"  # "vgg16", "resnet18", "efficientnet_b0", "lenet"

    # Obtem as transformações de pré-processamento para treino e teste
    train_transforms, test_transforms = ModelFactory.get_transforms(model_name)

    dataset_dir = 'espectrograms'
    subjects = os.listdir(dataset_dir)
    
    for i in range(len(subjects)):
        subjects_train = [s for s in subjects if s != f'{subjects[i]}']

        # Datasets com cache: imagens são carregadas e transformadas UMA vez e mantidas na RAM
        test_dataset = CachedImageFolder(
            datasets.ImageFolder(root=f'{dataset_dir}/{subjects[i]}', transform=test_transforms)
        )
        train_dataset = CachedConcatDataset(
            [datasets.ImageFolder(root=f'{dataset_dir}/{j}', transform=train_transforms) for j in subjects_train]
        )

        classifier = SleepStageClassifier(train_dataset=train_dataset, test_dataset=test_dataset, model_name=model_name)
        classifier.apply_epochs(epochs=20, directory='models', name=f'{model_name}_finetuned{i+1:02d}.pth', save_history=True)

        # Libera memória do fold anterior antes de carregar o próximo
        del classifier, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()











    #for model in os.listdir('models'):
    #    classifier.load_model(f'models/{model}')
    #    classifier.evaluate_model()