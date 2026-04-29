from spectrogram_converter import SpectrogramConverter
from data_splitter import DataSplitter
from model import SleepStageClassifier

if __name__ == "__main__":
    #converter = SpectrogramConverter()

    """
    converter.convert(
        edf_path="sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf",
        hypnogram_path="sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf"
        )
    """
    

    #splitter = DataSplitter()
    #splitter.split(seed=42)


    classifier = SleepStageClassifier()
    #classifier.apply_epochs(epochs=20, directory='models', name='vgg16_finetuned.pth', save_history=True)

    classifier.load_model('models/vgg16_finetuned.pth')
    classifier.evaluate_model()