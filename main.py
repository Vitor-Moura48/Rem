import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# Carrega os dados do arquivo EDF
raw = mne.io.read_raw_edf("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4002E0-PSG.edf", preload=True)

# lê as anotações do arquivo de hipnograma e as associa ao objeto Raw
annot = mne.read_annotations("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4002EC-Hypnogram.edf")
raw.set_annotations(annot)


# Exibe as informações do arquivo EDF
print(raw.info)
print(raw.get_channel_types())




"""
raw.plot(duration=10, n_channels=7, scalings="auto", show=True, block=True)
tfr.plot(dB=True, cmap="viridis")
"""



# Define os tipos de canais
raw.set_channel_types({
    'EEG Fpz-Cz': 'eeg',
    'EEG Pz-Oz': 'eeg',
    'EOG horizontal': 'eog',
    'Resp oro-nasal': 'misc',
    'EMG submental': 'emg',
    'Temp rectal': 'misc',
    'Event marker': 'stim'
})

# Seleciona apenas os sinais de EEG para análise
raw.pick('eeg')

# Re-referencia os sinais de EEG usando a média dos canais
raw.set_eeg_reference('average')

# Aplica um filtro passa-banda para remover ruídos e artefatos
raw.filter(1, 35)

annotation_desc_2_event_id = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 5,
    "Sleep stage R": 6,
}
# inverte o dicionário para mapear event_id para descrição
event_id_to_desc = {value: key for key, value in annotation_desc_2_event_id.items()}

# Extrai os eventos e as anotações do arquivo EDF para dados númericos
events, event_id = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30)

# Cria os epochs com base nos eventos e anotações
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=-1,
    tmax=30,
    baseline=(-1, 0),
    preload=True
)

start_epoch, range_epoch = 2700, 300
subset = epochs[start_epoch : start_epoch + range_epoch]

# Define a resolução de frequência para a análise de TFR
freqs = np.linspace(0.5, 30, 50)

# Configura a TFR
tfr = subset.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=freqs*0.5,
    average=False
)

# Aplica uma normalização para cada epoch
tfr.apply_baseline((-1, 0), mode="logratio")


#viridis, turbo, magma, plasma, inferno, greys, cmrmap, bynary, afmhot, hsv, hot, gray 


"""

# Cria o diretório se não existir
os.makedirs("espectrograms", exist_ok=True)

# Salva os espectrogramas para cada epoch e canal
for i in range(len(tfr)):
    for ch in tfr.ch_names:
        figs = tfr[i].plot(picks=[ch], show=False, cmap="turbo", vlim=(-0.5, 2), colorbar=False)

        # Obtém o event_id da epoch atual
        event_id = subset.events[i, 2]
        # Converte para o nome do estágio
        stage = event_id_to_desc[event_id]
        # Substitui espaços por underscores para o nome do arquivo
        stage_clean = stage.replace(" ", "_")

        for fig in figs:
            ax = fig.axes[0]
            ax.axis("off")

            fig.savefig(
                f"espectrograms/[{stage_clean}]_{ch}_epoch_{i + start_epoch}.png", 
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)
"""


tfr[1].plot(
    cmap="turbo",
    vlim=(-0.5, 2),
)