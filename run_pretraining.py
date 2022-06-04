import argparse
from cProfile import label
import os
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from transformers import AutoModelForAudioClassification
import torch
from torch import optim
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader
from custom_datasets.pretraining_dataset import pretraining_dataset

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
    "--model_name",
    help="Name of the pretrained model to be load",
    required=True)
parser.add_argument(
    "--output_checkpoint_folder",
    help="Output folder to store model checkpoints",
    default=None,
    required=True)
parser.add_argument(
    "--output_log_file",
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
parser.add_argument(
    "--n_epochs",
    help="Number of pretraining epochs",
    type=int,
    default=10,
    required=False)
parser.add_argument(
    "--learning_rate",
    help="Learning rate",
    type=float,
    default=1e-6,
    required=False)
parser.add_argument(
    "--step_size",
    help="Number of epochs before reducing the LR",
    type=int,
    default=20,
    required=False)
parser.add_argument(
    "--log_steps",
    help="Number of batch to process before printing info (such as current learning rate and loss value)",
    type=int,
    default=30,
    required=False)
neural_parser = parser.add_mutually_exclusive_group(required=False)
neural_parser.add_argument('--neural_augmentation', dest='neural_augmentation', action='store_true')
neural_parser.add_argument('--no-neural_augmentation', dest='neural_augmentation', action='store_false')
parser.set_defaults(neural_augmentation=True)
traditional_parser = parser.add_mutually_exclusive_group(required=False)
traditional_parser.add_argument('--traditional_augmentation', dest='traditional_augmentation', action='store_true')
traditional_parser.add_argument('--no-traditional_augmentation', dest='traditional_augmentation', action='store_false')
parser.set_defaults(traditional_augmentation=True)
args = parser.parse_args()

if not os.path.isdir(args.output_checkpoint_folder):
    os.mkdir(args.output_checkpoint_folder)

# device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model initialization
if args.model_name == 'wav2vec2':
    model_checkpoint = "facebook/wav2vec2-base"
elif args.model_name == 'wavlm':
    model_checkpoint = "microsoft/wavlm-base"  
else:
    print('Model Checkpoint not valid! Try again!')
    exit()
model = AutoModelForAudioClassification.from_pretrained(model_checkpoint, output_hidden_states=True, num_labels=6)
model.to(device)

# dataloader creation
train_df = pd.read_csv(os.path.join(args.csv_folder, "train.csv"))
le = preprocessing.LabelEncoder()
le.fit(train_df.label)
train_df['categorical_label'] = le.transform(train_df.label)
train_dataloader = pretraining_dataset(train_df, args.wav_folder, model_checkpoint, neural_augmentation=args.neural_augmentation, traditional_augmentation=args.traditional_augmentation)
train_dataloader = DataLoader(train_dataloader, shuffle=True, batch_size=args.batch_size, num_workers=args.n_workers)

dev_df = pd.read_csv(os.path.join(args.csv_folder, "devel.csv"))
le = preprocessing.LabelEncoder()
le.fit(dev_df.label)
dev_df['categorical_label'] = le.transform(dev_df.label)
dev_dataloader = pretraining_dataset(dev_df, args.wav_folder, model_checkpoint, neural_augmentation=args.neural_augmentation, traditional_augmentation=args.traditional_augmentation)
dev_dataloader = DataLoader(dev_dataloader, shuffle=True, batch_size=args.batch_size, num_workers=args.n_workers)



# training parameters configuration
step_size = args.step_size * len(train_dataloader)
loss_function = CosineEmbeddingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.5)

with open(args.output_log_file, "w") as f:
    f.write('Start Training!')

best_loss = None
best_epoch = None
for i in range(args.n_epochs): 
    print(f'Starting epoch {i + 1}')
    current_loss = 0.0
    model.train()

    for j, data in enumerate(tqdm(train_dataloader), 0):
        pair, label = data
        first_embeddings = torch.nn.functional.normalize(model(pair[0].to(device)).hidden_states[-1].mean(1), dim=-1)
        second_embeddings = torch.nn.functional.normalize(model(pair[1].to(device)).hidden_states[-1].mean(1), dim=-1)
        loss = loss_function(first_embeddings, second_embeddings, label.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        current_loss += loss.item()

        if (j+1)%args.log_steps == 0:
            print("LR:", scheduler.get_last_lr())
            print('Train loss at epoch %5d after mini-batch %5d: %.8f' % (i+1, j+1, current_loss/args.log_steps))
            with open(args.output_log_file, "a") as f:
                f.write('Train loss at epoch %5d after mini-batch %5d: %.8f\n' % (i+1, j+1, current_loss/args.log_steps))
            current_loss = 0.0

    model.eval()
    current_loss = 0.0
    with torch.no_grad():
        for j, data in enumerate(tqdm(dev_dataloader), 0):
            pair, label = data
            first_embeddings = torch.nn.functional.normalize(model(pair[0].to(device)).hidden_states[-1].mean(1), dim=-1)
            second_embeddings = torch.nn.functional.normalize(model(pair[1].to(device)).hidden_states[-1].mean(1), dim=-1)
            loss = loss_function(first_embeddings, second_embeddings, label.to(device))
            current_loss += loss.item()

        
    print('Dev loss at epoch %5d: %.8f' % (i+1, current_loss/len(dev_dataloader)))
    with open(args.output_log_file, "a") as f:
        f.write('Dev loss at epoch %5d: %.8f\n' % (i+1, current_loss/len(dev_dataloader)))
    if best_loss == None or best_loss > current_loss/len(dev_dataloader):
        best_loss = current_loss/len(dev_dataloader)
        best_epoch = i+1
        model_name = "model_" + str(i+1) + ".model"
        ckp_dir = args.output_checkpoint_folder + "/" + str(model_name) 
        torch.save(model, ckp_dir)

print('Training finished! Best model found at epoch %5d with a dev loss value of %.8f' % (best_epoch, best_loss))
with open(args.output_log_file, "a") as f:
    f.write('Training finished! Best model found at epoch %5d with a dev loss value of %.8f\n' % (best_epoch, best_loss))

#package_to_hub(model=model, # Our trained model
#               model_name=model_name, # The name of our trained model 
#               commit_message='gender_invariant_wav2vec2_base')
