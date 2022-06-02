# ComParE 2022 - Vocalisation Challange

Lorem ipsum

## Pretraining

Run pretraining:

```
python run_pretraining.py \
  --csv_folder data/dist/lab \
  --wav_folder data/dist/wav \
  --output_checkpoint_folder data/pretraining_checkpoints \
  --output_log_file data/pretraing_log.log \
  --batch_size 16 \
  --n_workers 4 \
  --n_epochs 100 \
  --learning_rate 1e-6 \
  --step_size 20 \
  --log_steps 20 \
  --neural_augmentation \
  --traditional_augmentation
```

## Finetuning

Run finetuning:

```
python run_pretraining.py \
  --csv_folder data/dist/lab \
  --batch_size 16 \
  --n_workers 4 \
  --n_epochs 20 \
  --learning_rate 3e-5 \
```
