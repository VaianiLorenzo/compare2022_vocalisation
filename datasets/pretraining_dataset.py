import os
import torch.utils.data as data
from transformers import Wav2Vec2FeatureExtractor
from random import randint
import librosa

class pretraining_dataset(data.Dataset):

    def __init__(self, train_df, wav_folder):
        self.files = list(train_df.filename)
        self.labels = list(train_df.label)
        self.categorical_labels = list(train_df.categorical_label)
        self.train_df = train_df
        self.wav_folder = wav_folder
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")


    def __getitem__(self, index):
        aug_type = randint(0,3)
        if aug_type < 2:
            label = 0
            while True:
                row = self.train_df.sample()
                if row.categorical_label.item() != self.categorical_labels[index]:
                    break
            filename = row.filename.item()
        elif aug_type == 2:
            label = 1
            filename = os.listdir("data/augmented/DL")[randint(0, len(os.listdir("data/augmented/DL"))-1)]
        else:
            label = 1
            filename = os.listdir("data/augmented/traditional")[randint(0, len(os.listdir("data/augmented/traditional"))-1)]
        
        data_mono, sr = librosa.load(os.path.join(self.wav_folder, self.files[index]), sr=16000, mono=True, res_type='soxr_qq')
        first = self.feature_extractor(data_mono, return_tensors="pt", sampling_rate=sr, padding="max_length", max_length=16000, truncation=True).input_values
        data_mono, sr = librosa.load(os.path.join(self.wav_folder, filename), sr=16000, mono=True, res_type='soxr_qq')
        second = self.feature_extractor(data_mono, return_tensors="pt", sampling_rate=sr, padding="max_length", max_length=16000, truncation=True).input_values
        return [first, second], label
            

    def __len__(self):
        return len(self.files)
