import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import json

class SpectrogramConverter:
    def __init__(
        self,
        output_dir="spectrograms",
        freqs=None,
        cmap="inferno",
        epoch_range=300,
        dpi=100,
        vlim=(-3.0, 3.0),
        global_stats_path_fpz="global_freq_stats_FpzCz.json",
        global_stats_path_pzoz="global_freq_stats_PzOz.json",
    ):
        self.output_dir = output_dir
        self.freqs = freqs if freqs is not None else np.linspace(0.5, 35, 90)
        self.cmap = cmap
        self.epoch_range = epoch_range
        self.dpi = dpi
        self.vlim = vlim

        self.freq_means = []
        self.freq_stds  = []

        for label, path in [("FpzCz", global_stats_path_fpz), ("PzOz", global_stats_path_pzoz)]:

            if path and os.path.exists(path):
                with open(path, "r") as f:
                    stats = json.load(f)
                self.freq_means.append(np.array(stats["means"]))
                self.freq_stds.append(np.array(stats["stds"]))
                print(f"Estatísticas [{label}] carregadas de {path}")
                
            else:
                self.freq_means.append(None)
                self.freq_stds.append(None)
                print(f"AVISO: {path} não encontrado. Z-score para [{label}] não será aplicado.")

    def convert(self, edf_path, hypnogram_path):

        raw = mne.io.read_raw_edf(edf_path, preload=True)

        annot = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annot)

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
        raw.pick(picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])

        raw.filter(0.5, 35.0, verbose=False)

        annotation_desc_2_event_id = {
            "Sleep stage W":  1,
            "Sleep stage 1": 2,
            "Sleep stage 2": 3,
            "Sleep stage 3": 4,
            "Sleep stage 4": 4,  # AASM
            "Sleep stage R":  5,
        }
        event_id_to_desc = {
            1: "Sleep stage W",
            2: "Sleep stage N1",
            3: "Sleep stage N2",
            4: "Sleep stage N3",
            5: "Sleep stage R",
        }

        events, event_id = mne.events_from_annotations(
            raw,
            event_id=annotation_desc_2_event_id,
            chunk_duration=30.0,
        )

        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=-1.0,
            tmax=30.0,
            preload=True,
            baseline=None
        )

        # Divide os epochs em blocos de x para evitar sobrecarga de memória
        splited_epochs = [epochs[i : i + self.epoch_range] 
                          for i in range(0, len(epochs), self.epoch_range)
                        ]

        # Cria o diretório se não existir
        os.makedirs(self.output_dir, exist_ok=True)

        # Extrai o ID do sujeito a partir do nome do arquivo EDF
        filename = os.path.basename(edf_path)
        subject_id = filename[3:5]
        # Cria um subdiretório para o sujeito
        os.makedirs(f"{self.output_dir}/subject_{subject_id}", exist_ok=True)

        # Cria subdiretórios para cada estágio
        for stage in event_id_to_desc.values():
            stage_clean = stage.replace(" ", "_")
            os.makedirs(f"{self.output_dir}/subject_{subject_id}/{stage_clean}", exist_ok=True)

        for subset in splited_epochs:

            # Configura a TFR — potência absoluta, sem baseline, sem logratio
            tfr = subset.compute_tfr(
                method="morlet",
                freqs=self.freqs,
                n_cycles=self.freqs*0.5,
                average=False,
                decim=8,
            )

            # Extrai a matriz de potência bruta: shape (n_epochs, n_channels, n_freqs, n_times)
            power_data = tfr.data
            power_data_db = 10 * np.log10(power_data + 1e-20)

            # Normaliza cada canal com suas próprias estatísticas
            normalized_data = np.empty_like(power_data_db)
            for ch_idx in range(2):
                if self.freq_means[ch_idx] is not None:
                    m = self.freq_means[ch_idx][np.newaxis, np.newaxis, :, np.newaxis]
                    s = self.freq_stds[ch_idx][np.newaxis, np.newaxis, :, np.newaxis]
                    normalized_data[:, ch_idx, :, :] = (power_data_db[:, ch_idx, :, :] - m) / s
                else:
                    normalized_data[:, ch_idx, :, :] = power_data_db[:, ch_idx, :, :]
        
            # Usa o vlim fixo definido no construtor. Agora ele representa Desvios Padrões.
            vmin, vmax = self.vlim

            # Salva os espectrogramas renderizando diretamente da matriz numérica
            for i in range(power_data.shape[0]):

                # Obtém o event_id da epoch atual
                epoch_event_id = subset.events[i, 2]

                # Converte para o nome do estágio
                stage = event_id_to_desc[epoch_event_id]

                # Substitui espaços por underscores para o nome do arquivo
                stage_clean = stage.replace(" ", "_")

                run_id = str(uuid.uuid4())[:8]
                base_path = f"{self.output_dir}/subject_{subject_id}/{stage_clean}/{run_id}"

                power_ch0 = normalized_data[i, 0, :, :]
                power_ch1 = normalized_data[i, 1, :, :]
                self._save_spectrogram_image(power_ch0, power_ch1, f"{base_path}.png", vmin, vmax)

            del tfr, power_data, power_data_db, normalized_data

    def _save_spectrogram_image(self, power_ch0, power_ch1, filepath, vmin, vmax):
            
        cmap = plt.colormaps[self.cmap]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        rgba_ch0 = cmap(norm(np.flipud(power_ch0)))  # (n_freqs, n_times, 4)
        rgba_ch1 = cmap(norm(np.flipud(power_ch1)))  # (n_freqs, n_times, 4)

        # Separador de 5 pixels pretos opacos
        n_times   = rgba_ch0.shape[1]
        separator = np.zeros((5, n_times, 4))
        separator[:, :, 3] = 1.0  # alpha = 1 (opaco)

        combined = np.vstack([rgba_ch0, separator, rgba_ch1])
        plt.imsave(filepath, combined, dpi=self.dpi)