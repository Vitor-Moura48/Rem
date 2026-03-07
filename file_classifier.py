import os
import shutil

for i in os.listdir("espectrograms"):
    splited = i.split("_EEG")

    os.makedirs("training/" + splited[0], exist_ok=True)

    # copia o arquivo para a nova pasta
    src = os.path.join("espectrograms", i)  # Caminho completo do arquivo original
    dst = os.path.join(splited[0], i)       # Caminho completo da nova pasta
    shutil.copy(
        src,
        dst
    )
    