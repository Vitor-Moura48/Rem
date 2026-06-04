import mne
import numpy as np
import os
import uuid
import json
from PIL import Image

class SpectrogramConverter:
    def __init__(
        self,
        output_dir="spectrograms",
        freqs=None,
        epoch_range=300,
        dpi=100,
        vlim=(-1.8, 2.2),
        vlim_eog=None,
        global_stats_path_fpz="global_freq_stats_FpzCz.json",
        global_stats_path_pzoz="global_freq_stats_PzOz.json",
        global_stats_path_eog="global_freq_stats_EOG.json",
        save_mode="rgb",
    ):
        self.output_dir  = output_dir
        self.freqs       = freqs if freqs is not None else np.linspace(0.5, 35, 90)
        self.epoch_range = epoch_range
        self.dpi         = dpi
        self.vlim        = vlim
        self.vlim_eog    = vlim_eog if vlim_eog is not None else vlim
        self.save_mode   = save_mode

        self.freq_means = []
        self.freq_stds  = []

        stat_files = [
            ("FpzCz", global_stats_path_fpz),
            ("PzOz",  global_stats_path_pzoz),
            ("EOG",   global_stats_path_eog),
        ]
        for label, path in stat_files:
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

        # Seleciona EEG + EOG para análise
        raw.pick(picks=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'])

        # Mesmo filtro para os 3 canais (mantém espectrogramas com mesma dimensão)
        raw.filter(0.5, 35.0, verbose=False)

        annotation_desc_2_event_id = {
            "Sleep stage W":  1,
            "Sleep stage 1":  2,
            "Sleep stage 2":  3,
            "Sleep stage 3":  4,
            "Sleep stage 4":  4,  # AASM
            "Sleep stage R":  5,
        }
        event_id_to_desc = {
            1: "Sleep_stage_W",
            2: "Sleep_stage_N1",
            3: "Sleep_stage_N2",
            4: "Sleep_stage_N3",
            5: "Sleep_stage_R",
        }

        events, event_id = mne.events_from_annotations(
            raw,
            event_id=annotation_desc_2_event_id,
            chunk_duration=30.0,
        )
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=-1.0, tmax=30.0,
            preload=True,
            baseline=None,
        )

        splited_epochs = [epochs[i : i + self.epoch_range]
                          for i in range(0, len(epochs), self.epoch_range)]

        os.makedirs(self.output_dir, exist_ok=True)
        filename   = os.path.basename(edf_path)
        subject_id = filename[3:5]
        subj_dir   = f"{self.output_dir}/subject_{subject_id}"
        os.makedirs(subj_dir, exist_ok=True)

        for stage in event_id_to_desc.values():
            os.makedirs(f"{subj_dir}/{stage}", exist_ok=True)

        for subset in splited_epochs:

            tfr = subset.compute_tfr(
                method="morlet",
                freqs=self.freqs,
                n_cycles=self.freqs * 0.5,
                average=False,
                decim=8,
                picks=['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'],
            )

            power_data = tfr.data                           # (n_epochs, 3, n_freqs, n_times)
            power_db   = 10 * np.log10(power_data + 1e-20)

            # Normaliza cada canal com suas próprias estatísticas (Z-score por frequência)
            normalized = np.empty_like(power_db)
            for ch_idx in range(3):
                if self.freq_means[ch_idx] is not None:
                    m = self.freq_means[ch_idx][np.newaxis, np.newaxis, :, np.newaxis]
                    s = self.freq_stds[ch_idx][np.newaxis, np.newaxis, :, np.newaxis]
                    normalized[:, ch_idx, :, :] = (power_db[:, ch_idx, :, :] - m) / s
                else:
                    normalized[:, ch_idx, :, :] = power_db[:, ch_idx, :, :]

            for i in range(power_data.shape[0]):
                epoch_event_id = subset.events[i, 2]
                stage          = event_id_to_desc[epoch_event_id]
                run_id         = str(uuid.uuid4())[:8]
                filepath       = f"{subj_dir}/{stage}/{run_id}.png"

                ch0 = normalized[i, 0, :, :]
                ch1 = normalized[i, 1, :, :]
                ch2 = normalized[i, 2, :, :]
                self._save_image(ch0, ch1, ch2, filepath)

            del tfr, power_data, power_db, normalized

    def _save_image(self, ch0, ch1, ch2, filepath):
        vmin, vmax         = self.vlim
        vmin_eog, vmax_eog = self.vlim_eog

        def to_uint8(mat, lo, hi):
            """Normaliza uma matriz para [0, 255] com base nos limites fornecidos."""
            clipped = np.clip(mat, lo, hi)
            scaled  = (clipped - lo) / (hi - lo) * 255
            return np.flipud(scaled).astype(np.uint8)

        r = to_uint8(ch0, vmin, vmax)
        g = to_uint8(ch1, vmin, vmax)
        b = to_uint8(ch2, vmin_eog, vmax_eog)   # EOG usa vlim próprio

        if self.save_mode == "rgb":
            # Canal R=EEG Fpz-Cz | G=EEG Pz-Oz | B=EOG — imagem RGB compacta
            img_array = np.stack([r, g, b], axis=-1)  # (H, W, 3)
        else:
            # Modo "stacked": canais empilhados verticalmente com separador preto
            n_times   = r.shape[1]
            separator = np.zeros((5, n_times), dtype=np.uint8)
            gray = np.vstack([r, separator, g, separator, b])  # (3H+10, W)
            img_array = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3) grayscale

        Image.fromarray(img_array).save(filepath)