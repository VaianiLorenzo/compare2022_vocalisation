import os
import torch.utils.data as data
from transformers import Wav2Vec2FeatureExtractor
from random import randint
import librosa
import enum
import numpy as np

class PairType(enum.Enum):
    neural_augmentation = 1
    feature_augmentation = 2
    negative_sample = 3

class pretraining_dataset(data.Dataset):

    def __init__(self, 
                train_df, 
                wav_folder, 
                traditional_augmented_folder="data/augmented/traditional", 
                neural_augmented_path="data/augmented/neural/M1"):
        self.files = list(train_df.filename)
        self.labels = list(train_df.label)
        self.categorical_labels = list(train_df.categorical_label)
        self.train_df = train_df
        self.wav_folder = wav_folder
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.traditional_augmented_folder = traditional_augmented_folder
        self.neural_augmented_path = neural_augmented_path


    def __getitem__(self, index):
        sampling_type_list = [PairType.neural_augmentation, PairType.feature_augmentation, PairType.negative_sample]
        sampling_type = np.random.choice(sampling_type_list, 1, p=[0.25, 0.25, 0.5])[0]
        if sampling_type == PairType.negative_sample:
            label = 0
            while True:
                row = self.train_df.sample()
                if row.categorical_label.item() != self.categorical_labels[index]:
                    break
            filename = row.filename.item()
        elif sampling_type == PairType.neural_augmentation:
            label = 1
            filename = os.listdir(self.neural_augmented_path)[randint(0, len(os.listdir(self.neural_augmented_path))-1)]
        elif PairType.feature_augmentation:
            label = 1
            filename = os.listdir(self.traditional_augmented_folder)[randint(0, len(os.listdir(self.traditional_augmented_folder))-1)]
        else:
            # error in augmentation type, return the same with label 1.
            label = 1
            filename = os.path.join(self.wav_folder, self.files[index])

        
        data_mono, sr = librosa.load(os.path.join(self.wav_folder, self.files[index]), sr=16000, mono=True, res_type='soxr_qq')
        first = self.feature_extractor(data_mono, return_tensors="pt", sampling_rate=sr, padding="max_length", max_length=16000, truncation=True).input_values
        data_mono, sr = librosa.load(os.path.join(self.wav_folder, filename), sr=16000, mono=True, res_type='soxr_qq')
        second = self.feature_extractor(data_mono, return_tensors="pt", sampling_rate=sr, padding="max_length", max_length=16000, truncation=True).input_values
        return [first, second], label
            

    def __len__(self):
        return len(self.files)
