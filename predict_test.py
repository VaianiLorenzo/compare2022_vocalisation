import argparse
from cProfile import label
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForAudioClassification
from transformers import AutoFeatureExtractor
import torch
from torch.utils.data import DataLoader
from custom_datasets.finetuning_dataset import finetuning_dataset

parser = argparse.ArgumentParser(description="WavLM pretaraining with contrastive learning")
parser.add_argument(
    "--csv_folder",
    help="Input folder containing the csv files of each split",
    required=True)
parser.add_argument(
    "--wav_folder",
    help="Input folder containing the wav files",
    required=True)
parser.add_argument(
    "--checkpoint_path",
    help="Name of the pretrained model to be load",
    required=True)
parser.add_argument(
        "--feature_extractor",
        help="Feature extractor checkpoint to be loaded",
        required=True)
parser.add_argument(
    "--output_file",
    help="Name to assign to the output log file",
    default=None,
    required=True)
parser.add_argument(
    "--batch_size",
    help="Training batch size",
    type=int,
    default=16,
    required=False)
parser.add_argument(
    "--n_workers",
    help="Number of workers",
    type=int,
    default=4,
    required=False)
args = parser.parse_args()

# device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model initialization
model = AutoModelForAudioClassification.from_pretrained(args.checkpoint_path, local_files_only=True)
model.to(device)
model.eval()

if args.feature_extractor == 'wav2vec2':
    model_checkpoint = "facebook/wav2vec2-base"
elif args.feature_extractor == 'wavlm':
    model_checkpoint = "microsoft/wavlm-base"  
else:
    print('Feature Extractor Checkpoint not valid! Try again!')
    exit()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

train_df = pd.read_csv(os.path.join(args.csv_folder, "train.csv"))
emotions = train_df['label'].unique()
label2id, id2label = dict(), dict()
for i, label in enumerate(emotions):
    label2id[label] = str(i)
    id2label[str(i)] = label

# dataloader creation
test_df = pd.read_csv(os.path.join(args.csv_folder, "test.csv"))
test_dataloader = finetuning_dataset(test_df, args.wav_folder, feature_extractor, 2, device, neural_augmentation=False, traditional_augmentation=False)
test_dataloader = DataLoader(test_dataloader, shuffle=True, batch_size=args.batch_size, num_workers=args.n_workers)

predicted = []
files = []
for j, data in enumerate(tqdm(test_dataloader), 0):
    predictions = model(data["input_values"].to(device))
    for p,f in zip(predictions["logits"], data["filename"]):
        files.append(f)
        predicted.append(id2label[str(np.argmax(p.cpu().detach().numpy()))])

predictions_df = pd.DataFrame()
predictions_df["filename"] = files
predictions_df["prediction"] = predicted
predictions_df = predictions_df.sort_values(by=["filename"])
predictions_df.to_csv("data/" + args.output_file, index=False)