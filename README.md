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
  --model_name wav2vec2 \
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
python run_finetuning.py \
  --csv_folder data/dist/lab \
  --wav_folder data/dist/wav \
  --output_dir wav2vec2_pretrainingmode_finetuningmode \
  --pretrained_model data/wav2vec2_traditional_pretraining_checkpoints/model_x.model \
  --feature_extractor wav2vec2 \
  --batch_size 16 \
  --n_workers 4 \
  --n_epochs 200 \
  --learning_rate 3e-5 \
  --neural_augmentation   \
  --traditional_augmentation \
  --no-valid_aug
```

## Test Prediction

Run test predictions:
```
python predict_test.py 
  --csv_folder data/dist/lab \
  --wav_folder data/dist/wav \
  --checkpoint_path data/wavlm_trad_trad_finetuned/checkpoint-4800 \
  --feature_extractor wavlm \
  --output_file wavlm_traditional_traditional_predictions.csv \
  --batch_size 4 \
  --n_workers 4
```
