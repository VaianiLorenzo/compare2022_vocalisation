import argparse
from cProfile import label
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
    default=39,
    required=False)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
model.to(device)
train_dataloader = torch.load(os.path.join(args.dataloader_folder, "pretrain_dataloader.bkp"))

step_size = args.step_size * len(train_dataloader)
loss_function = CosineEmbeddingLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma = 0.5)

for i in range(args.n_epochs): 
    print(f'Starting epoch {i + 1}')
    current_loss = 0.0
    model.train()

    for j, data in enumerate(tqdm(train_dataloader), 0):
        pair, label = data
        first_embeddings = torch.nn.functional.normalize(model(pair[0].to(device)).embeddings, dim=-1)
        second_embeddings = torch.nn.functional.normalize(model(pair[1].to(device)).embeddings, dim=-1)
        loss = loss_function(first_embeddings, second_embeddings, label.to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        current_loss += loss.item()

        if j % args.log_steps == 0 and j != 0:
            print("LR:", scheduler.get_last_lr())
            print('Train loss at epoch %5d after mini-batch %5d: %.8f' % (i+1, j+1, current_loss / args.log_steps))
            with open(args.output_log_file, "a") as f:
                f.write('Train loss at epoch %5d after mini-batch %5d: %.8f\n' % (i+1, j+1, current_loss / args.log_steps))
            current_loss = 0.0

    model_name = "model_" + str(i+1) + ".model"
    ckp_dir = args.output_checkpoint_folder + "/" + str(model_name) 
    torch.save(model, ckp_dir)
