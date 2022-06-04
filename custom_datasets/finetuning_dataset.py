import os
import torch
import torchaudio
import torch.utils.data as data
import librosa
import enum
from random import randint
import numpy as np

class PairType(enum.Enum):
    neural_augmentation = 1
    feature_augmentation = 2
    negative_sample = 3

class finetuning_dataset(data.Dataset):
    def __init__(self, examples, wav_folder, feature_extractor, max_duration, device, 
                traditional_augmented_folder="data/augmented/traditional", 
                neural_augmented_folder="data/augmented/neural",
                neural_augmentation=True,
                traditional_augmentation=True):
        self.files = list(examples['filename'])
        self.labels = list(examples['label'])
        self.wav_folder = wav_folder
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.device = device
        self.traditional_augmented_folder = traditional_augmented_folder
        self.neural_augmented_folder = neural_augmented_folder
        if neural_augmentation:
            if traditional_augmentation:
                self.proportions = [0.25, 0.25, 0.5]
            else:
                self.proportions = [0.5, 0.0, 0.5]
        else:
            if traditional_augmentation:
                self.proportions = [0.0, 0.5, 0.5]
            else:
                self.proportions = [0.0, 0.0, 1.0]

    def __getitem__(self, index):

        sampling_type_list = [PairType.neural_augmentation, PairType.feature_augmentation, PairType.negative_sample]
        sampling_type = np.random.choice(sampling_type_list, 1, p=self.proportions)[0]
        if sampling_type == PairType.negative_sample:
            filename = os.path.join(self.wav_folder, self.files[index])
        elif sampling_type == PairType.neural_augmentation:
            filename = os.path.join(self.neural_augmented_folder, self.files[index])
        elif PairType.feature_augmentation:
            filename = os.path.join(self.traditional_augmented_folder, self.files[index])
        else:
            # error in augmentation type, return the same with label 1.
            filename = os.path.join(self.wav_folder, self.files[index])

        inputs = self.feature_extractor(
            librosa.resample(np.asarray(torchaudio.load(filename)[0]), 48_000, 16_000).squeeze(0),
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
            truncation=True,
            padding='max_length'
        )#.to(self.device)
        item = {'input_values': inputs['input_values'].squeeze(0)}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.files)