import os

from spectrogram_converter import SpectrogramConverter
from data_splitter import DataSplitter
from model import SleepStageClassifier
from torchvision import datasets
from torch.utils.data import ConcatDataset

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


    # Obtem as transformações de pré-processamento para treino e teste
    train_transforms, test_transforms = SleepStageClassifier.get_transforms()

    dataset_dir = 'espectrograms'
    subjects = os.listdir(dataset_dir)
    
    for i in range(len(subjects)):
        subjects_train = [s for s in subjects if s != f'{subjects[i]}']

        test_dataset = datasets.ImageFolder(root=f'{dataset_dir}/{subjects[i]}', transform=test_transforms)
        train_dataset = ConcatDataset([datasets.ImageFolder(root=f'{dataset_dir}/{j}', transform=train_transforms) for j in subjects_train])
        
        # Adiciona os atributos de samples e classes ao dataset de treino para compatibilidade com o método de treinamento
        train_dataset.samples = [s for ds in train_dataset.datasets for s in ds.samples]
        train_dataset.classes = train_dataset.datasets[0].classes

        classifier = SleepStageClassifier(train_dataset=train_dataset, test_dataset=test_dataset)
        classifier.apply_epochs(epochs=20, directory='models', name=f'vgg16_finetuned{i+1:02d}.pth', save_history=True)

    for model in os.listdir('models'):
        classifier.load_model(f'models/{model}')
        classifier.evaluate_model()