# ComParE 2022 - Vocalisation Challange

Lorem ipsum

## Pretraining

Create pretraining dataloader:

```
python pretraining_dataloader_creation.py \
  --csv_folder data/dist/lab \
  --wav_folder data/dist/wav \
  --output_folder data/dataloaders
```

Run pretraining:

```
python run_pretraining.py \
  --dataloader_folder data/dataloaders \
  --batch_size 16 \
  --n_epochs 10 \
  --learning_rate 1e-5
```
