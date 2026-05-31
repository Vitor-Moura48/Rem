import os
import random
import shutil

class DataSplitter:
    def __init__(self, source_dir="espectrograms", dest_dir="dataset", train_ratio=0.7):

        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.train_ratio = train_ratio

    def split(self, seed=42):
    
        random.seed(seed)
        
        # Cria os diretórios de treino e teste
        os.makedirs(self.dest_dir, exist_ok=True)

        # Itera sobre as pastas de classificação
        for stage in os.listdir(self.source_dir):
            stage_path = os.path.join(self.source_dir, stage)

            if not os.path.isdir(stage_path):
                print(f"Aviso: {stage_path} não é um diretório. Pulando.")
                continue

            files = os.listdir(stage_path)
            random.shuffle(files)

            train_amount = int(self.train_ratio * len(files))
            splits = {
                "train": files[:train_amount],
                "test": files[train_amount:]
            }

            print(f"{stage}: {len(files)} arquivos → {train_amount} treino, {len(files) - train_amount} teste")

            for split_name, split_files in splits.items():
                out_dir = os.path.join(self.dest_dir, split_name, stage)
                os.makedirs(out_dir, exist_ok=True)

                for file in split_files:
                    shutil.copy(
                        os.path.join(stage_path, file),
                        os.path.join(out_dir, file)
                    )