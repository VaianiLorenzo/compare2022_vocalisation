# Transformer-based Non-Verbal Emotion Recognition: Exploring Model Portability across Speakers' Genders

This repository contains the code for the paper "Transformer-based Non-Verbal Emotion Recognition: Exploring Model Portability across Speakers’ Genders" submitted for the [ComParE 2022](http://www.compare.openaudio.eu/2022-2/) Vocalisation Sub-Challenge.

Recognizing emotions in non-verbal audio tracks requires a deep understanding of their underlying features. Traditional classifiers relying on excitation, prosodic, and vocal traction features are not always capable of effectively generalizing across speakers' genders. In the ComParE 2022 vocalisation sub-challenge we explore the use of a Transformer architecture trained on contrastive audio examples. We leverage augmented data to learn robust non-verbal emotion classifiers. We also investigate the impact of different audio transformations, including neural voice conversion, on the classifier capability to generalize across speakers' genders. The empirical findings indicate that neural voice conversion is beneficial in the pretraining phase, yielding an improved model generality, whereas is harmful at the finetuning stage as hinders model specialization for the task of non-verbal emotion recognition.

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

## Citation
```
@inproceedings{10.1145/3551876.3554801,
  author = {Vaiani, Lorenzo and Koudounas, Alkis and La Quatra, Moreno and Cagliero, Luca and Garza, Paolo and Baralis, Elena},
  title = {Transformer-Based Non-Verbal Emotion Recognition: Exploring Model Portability across Speakers' Genders},
  year = {2022},
  isbn = {9781450394840},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3551876.3554801},
  doi = {10.1145/3551876.3554801},
  booktitle = {Proceedings of the 3rd International on Multimodal Sentiment Analysis Workshop and Challenge},
  pages = {89–94},
  numpages = {6},
  keywords = {data augmentation, contrastive learning, non-verbal emotion recognition, audio classification},
  location = {Lisboa, Portugal},
  series = {MuSe' 22}
}
```
