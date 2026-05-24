import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

class SpectrogramConverter:
    def __init__(
        self, 
        output_dir="espectrograms", 
        freqs=None, vlim=(-0.5, 2), 
        cmap="turbo", epoch_range=300
    ):

        self.output_dir = output_dir
        self.freqs = freqs if freqs is not None else np.linspace(0.5, 30, 50) # Frequências de interesse
        self.vlim = vlim # Limites para a escala de cores dos espectrogramas
        self.cmap = cmap
        self.epoch_range = epoch_range

    def convert(self, edf_path, hypnogram_path):

        # Carrega os dados do arquivo EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True)

        # lê as anotações do arquivo de hipnograma e as associa ao objeto Raw
        annot = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annot)

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
        print(raw.info)
        
        # Seleciona apenas os sinais de EEG para análise
        raw.pick('eeg')

        # Re-referencia os sinais de EEG usando a média dos canais
        raw.set_eeg_reference('average')

        # Aplica um filtro passa-banda para remover ruídos e artefatos
        raw.filter(1, 35)

        # Cria um RawArray de canal único com a média dos canais EEG
        data = raw.get_data()                          # (x, n_samples)
        fused = data.mean(axis=0, keepdims=True)       # (1, n_samples)
        info = mne.create_info(ch_names=['EEG_fused'], sfreq=raw.info['sfreq'], ch_types=['eeg'])
        raw = mne.io.RawArray(fused, info)
        raw.set_annotations(annot)

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

        # Divide os epochs em blocos de 300 para evitar sobrecarga de memória
        splited_epochs = [epochs[i : i + self.epoch_range] 
                          for i in range(0, len(epochs), self.epoch_range)
                        ]

        # Cria o diretório se não existir
        os.makedirs(self.output_dir, exist_ok=True)

        # Cria subdiretórios para cada estágio
        for stage in event_id_to_desc.values():
            stage_clean = stage.replace(" ", "_")
            os.makedirs(f"{self.output_dir}/{stage_clean}", exist_ok=True)

        for subset in splited_epochs:

            # Configura a TFR
            tfr = subset.compute_tfr(
                method="morlet",
                freqs=self.freqs,
                n_cycles=self.freqs*0.5,
                average=False
            )

            # Aplica uma normalização para cada epoch
            tfr.apply_baseline((-1, 0), mode="logratio")

            # Salva os espectrogramas para cada epoch e canal
            for i in range(len(tfr)):
                for ch in tfr.ch_names:
                    figs = tfr[i].plot(picks=[ch], show=False, cmap=self.cmap, vlim=self.vlim, colorbar=False)

                    # Obtém o event_id da epoch atual
                    epoch_event_id = subset.events[i, 2]
                    # Converte para o nome do estágio
                    stage = event_id_to_desc[epoch_event_id]
                    # Substitui espaços por underscores para o nome do arquivo
                    stage_clean = stage.replace(" ", "_")

                    for fig in figs:
                        ax = fig.axes[0]
                        ax.axis("off")
                        
                        run_id = str(uuid.uuid4())[:8]
                        fig.savefig(
                            f"{self.output_dir}/{stage_clean}/{ch}_{run_id}.png", 
                            dpi=100,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close(fig)
                    
            del tfr # Libera a memória após processar cada bloco de epochs