import argparse
import os
from tqdm import tqdm
from transformers import WavLMForXVector
import torch
from torch import optim
from torch.nn import CosineEmbeddingLoss

parser = argparse.ArgumentParser(description="Train Perceiver")
parser.add_argument(
    "--dataloader_folder",
    help="Input folder containing the pretraining dataloader",
    required=True)
parser.add_argument(
    "--batch_size",
    help="Training batch size",
    type=int,
    default=16,
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
    default=1e-5,
    required=False)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
model.to(device)
train_dataloader = torch.load(os.path.join(args.dataloader_folder, "pretrain_dataloader.bkp"))

step_size = args.n_epochs * len(train_dataloader)
loss_function = CosineEmbeddingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.1)

for epoch in range(args.n_epochs): 
    print(f'Starting epoch {epoch + 1}')
    current_loss = 0.0
    model.train()

    for i, data in enumerate(tqdm(train_dataloader), 0):
        pair, label = data
        first = torch.stack([e[0].to(device) for e in pair]).to(device)
        second = torch.stack([e[1].to(device) for e in pair]).to(device)
        first_embeddings = torch.nn.functional.normalize(model(first).embeddings, dim=-1)
        second_embeddings = torch.nn.functional.normalize(model(second).embeddings, dim=-1)
        loss = loss_function(first_embeddings, second_embeddings, label)
        
        exit()
