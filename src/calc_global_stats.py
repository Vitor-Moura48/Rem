import mne
import numpy as np
import os
import json

def calculate_global_stats(paths, freqs):
    print("Iniciando cálculo das estatísticas globais por frequência...")

    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    n_channels = len(channels)
    
    all_power_sums = np.zeros((n_channels, len(freqs)))
    all_power_sq_sums = np.zeros((n_channels, len(freqs)))
    total_points = 0
    
    for edf_path, hypnogram_path in paths:
        print(f"Processando {os.path.basename(edf_path)}...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        annot = mne.read_annotations(hypnogram_path)
        raw.set_annotations(annot)
        raw.pick(picks=channels)
        raw.filter(0.5, 35, verbose=False)
        
        annotation_desc_2_event_id = {
            "Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
            "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5,
        }
        events, event_id = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30, verbose=False)
        
        epochs = mne.Epochs(raw, events, event_id, tmin=-1, tmax=30, preload=True, baseline=None, verbose=False)
        
        # Processa em blocos (chunks) para evitar MemoryError (igual no conversor)
        splited_epochs = [epochs[i : i + 300] for i in range(0, len(epochs), 300)]
        
       
        for subset in splited_epochs:
            # Configura a TFR
            tfr = subset.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs*0.5, average=False, verbose=False)
            power_data = tfr.data # shape: (n_epochs_chunk, n_channels, n_freqs, n_times)

            # Converte para dB
            power_db = 10 * np.log10(power_data + 1e-20)
        
            # Acumula as somas
            for ch_idx in range(n_channels):
                for f_idx in range(len(freqs)):
                    freq_data = power_db[:, ch_idx, f_idx, :]
                    all_power_sums[ch_idx, f_idx] += freq_data.sum()
                    all_power_sq_sums[ch_idx, f_idx] += (freq_data ** 2).sum()
                
            total_points += power_db.shape[0] * power_db.shape[3]
            del tfr, power_data, power_db
            
        
    print("\nCalculando Médias e Desvios Padrões finais...")
    means = all_power_sums / total_points
    variances = (all_power_sq_sums / total_points) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 1e-10))
    
    output_names = ["global_freq_stats_FpzCz.json", "global_freq_stats_PzOz.json"]
    for ch_idx, output_file in enumerate(output_names):

        stats = {
            "freqs": freqs.tolist(),
            "means": means[ch_idx].tolist(),
            "stds": stds[ch_idx].tolist()
        }
        
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=4)
        
        print(f"  Salvo: {output_file}")

if __name__ == "__main__":
    # Caminhos do dataset local para calcular a referência
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
        ("sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4151E0-PSG.edf", "sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4151EC-Hypnogram.edf")
    ]
    freqs = np.linspace(0.5, 35, 90)
    calculate_global_stats(paths, freqs)
