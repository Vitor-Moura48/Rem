import os
import gc
import torch

from spectrogram_converter import SpectrogramConverter
from data_splitter import DataSplitter
from model import SleepStageClassifier
from torchvision import datasets
from cached_dataset import CachedImageFolder, CachedConcatDataset
from model_factory import ModelFactory


def run_kfold(model_name, subjects_per_fold, dataset_dir='spectrograms', max_oversample=1):

    train_transforms, test_transforms = ModelFactory.get_transforms(model_name)

    subjects = sorted(os.listdir(dataset_dir))
    n = len(subjects)
    groups = [subjects[i : i + subjects_per_fold] for i in range(0, n, subjects_per_fold)]

    print(f"\n{'='*60}")
    print(f"  K-Fold Leave-Group-Out")
    print(f"  Sujeitos totais : {n}")
    print(f"  Sujeitos/Fold   : {subjects_per_fold}")
    print(f"  Nº de Folds     : {len(groups)}")
    print(f"{'='*60}\n")

    for fold_idx, test_group in enumerate(groups):

        train_subjects = [s for s in subjects if s not in test_group]

        print(f"[Fold {fold_idx+1:02d}/{len(groups)}]")
        print(f"  Teste  : {test_group}")
        print(f"  Treino : {train_subjects}\n")

        train_dataset = CachedConcatDataset(
            [datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=train_transforms) for s in train_subjects],
            augmentations=ModelFactory.get_augmentations(),
            max_oversample=max_oversample
        )
        test_dataset = CachedImageFolder(
            datasets.ImageFolder(root=f'{dataset_dir}/{test_group[0]}', transform=test_transforms)
        ) if subjects_per_fold == 1 else CachedConcatDataset(
            [datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_transforms) for s in test_group],
            augmentations=None,
            balance=False
        )

        fold_metrics_dir = f'metrics/{model_name}_fold{fold_idx+1:02d}'

        classifier = SleepStageClassifier(train_dataset=train_dataset, test_dataset=test_dataset, model_name=model_name)
        classifier.apply_epochs(
            epochs=30,
            directory='models',
            name=f'{model_name}_finetuned{fold_idx+1:02d}.pth',
            save_history=True,
            history_dir=fold_metrics_dir
        )

        classifier.load_model(f'models/{model_name}_finetuned{fold_idx+1:02d}.pth')
        print(f"\n[Avaliando {model_name} no fold {fold_idx+1}]")
        classifier.evaluate_model(save_dir=fold_metrics_dir)

        del classifier, train_dataset, test_dataset
        gc.collect()
        torch.cuda.empty_cache()


def run_manual(model_name, train_idx, test_idx, dataset_dir='spectrograms', max_oversample=1):

    train_transforms, test_transforms = ModelFactory.get_transforms(model_name)

    all_subjects = sorted(os.listdir(dataset_dir))
    train_subjects = [all_subjects[i] for i in train_idx]
    test_subjects  = [all_subjects[i] for i in test_idx]

    print(f"\n{'='*60}")
    print(f"  Manual")
    print(f"  Treino : {train_subjects}")
    print(f"  Teste  : {test_subjects}")
    print(f"{'='*60}\n")

    train_dataset = CachedConcatDataset(
        [datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=train_transforms) for s in train_subjects],
        augmentations=ModelFactory.get_augmentations(),
        max_oversample=max_oversample
    )
    test_dataset = CachedConcatDataset(
        [datasets.ImageFolder(root=f'{dataset_dir}/{s}', transform=test_transforms) for s in test_subjects],
        augmentations=None,
        balance=False
    )

    classifier = SleepStageClassifier(train_dataset=train_dataset, test_dataset=test_dataset, model_name=model_name)
    classifier.apply_epochs(
        epochs=30,
        directory='models',
        name=f'{model_name}_manual.pth',
        save_history=True,
        history_dir=f'metrics/{model_name}_manual'
    )

    classifier.load_model(f'models/{model_name}_manual.pth')
    print(f"\n[Avaliando {model_name} - manual]")
    classifier.evaluate_model(save_dir=f'metrics/{model_name}_manual')

    del classifier, train_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    
    """
    # (-3, 3) é o padrão do Z-Score. Apertar para (-2, 2) aumenta o contraste.
    vlim = (-1.8, 2.2)

    converter = SpectrogramConverter(
        output_dir='spectrograms',
        vlim=vlim,
    )

    paths = [
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4011E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4011EH-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4021E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4021EH-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4031E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4031EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4041E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4041EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4051E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4051EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4061E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4061EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4071E0-PSG.edf", "sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4071EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4081E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4081EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4091E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4091EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4101E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4101EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4111E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4111EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4121E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4121EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4131E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4131EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4141E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4141EU-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4151E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4151EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4161E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4161EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4171E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4171EU-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4181E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4181EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4191E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4191EP-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4201E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4201EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4211E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4211EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4221E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4221EJ-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4231E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4231EJ-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4241E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4241EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4251E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4251EP-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4261F0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4261FM-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4271F0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4271FC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4281G0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4281GC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4291G0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4291GA-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4301E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4301EC-Hypnogram.edf"),
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4311E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4311EC-Hypnogram.edf"),
    ]
    
    for edf_path, hypnogram_path in paths:
        converter.convert(edf_path=edf_path, hypnogram_path=hypnogram_path)
    """


    # Divide os dados em treino e teste (Não utilizado ao aplicar o LOSO)
    #splitter = DataSplitter()
    #splitter.split(seed=42)


    
    model_name = "lenet"

    # Métodos de execução
    # =================================================
    run_kfold(
        model_name=model_name,
        subjects_per_fold=8,
        max_oversample=2
        )
    #run_manual(
    #    model_name=model_name,
    #    train_idx=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #    test_idx=[6],  # 2, *6, 7, 9, 10*, 13
    #    max_oversample=1
    #    )
    

    