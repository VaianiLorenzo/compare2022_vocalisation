import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoFeatureExtractor, EarlyStoppingCallback, AutoModelForAudioClassification
import numpy as np 
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score
import librosa
import torch.nn as nn
from sklearn.utils import class_weight
import argparse
from tqdm import tqdm
from custom_datasets.finetuning_dataset import finetuning_dataset

import warnings
warnings.filterwarnings("ignore")

path = 'data/dist/wav/'

""" Trainer Class """
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").long()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    if args.feature_extractor == 'wav2vec2' or args.feature_extractor == 'wavlm':
        preds = np.argmax(pred.predictions, axis=1)
    else:
        preds = np.argmax(pred.predictions[0], axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    uar = recall_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    # print('F1 score: ' + str(f1))
    # print('Accuracy: ' + str(acc))
    # print('Precision: ' + str(precision))
    # print('Recall: ' + str(recall))
    print('UAR: ' + str(uar))
      
    return {
        'accuracy': acc,
        'uar': uar,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="WavLM finetuning")
    parser.add_argument(
        "--csv_folder",
        help="Input folder containing the csv files of each split",
        required=True)
    parser.add_argument(
        "--wav_folder",
        help="Input folder containing the wav files",
        required=True)
    parser.add_argument(
        "--pretrained_model",
        help="Pretrained model to be finetuned",
        required=True)
    parser.add_argument(
        "--feature_extractor",
        help="Feature extractor checkpoint to be loaded",
        required=True)
    parser.add_argument(
        "--batch_size",
        help="Batch size",
        default=16, 
        type=int,
        required=False)
    parser.add_argument(
        "--n_workers",
        help="Number of workers",
        type=int,
        default=4,
        required=False)
    parser.add_argument(
        "--n_epochs",
        help="Number of finetuning epochs",
        type=int,
        default=20,
        required=False)
    parser.add_argument(
        "--learning_rate",
        help="Learning rate",
        type=float,
        default=3e-5,
        required=False)
    parser.add_argument(
        "--output_dir",
        help="Output Directory",
        required=True)
    neural_parser = parser.add_mutually_exclusive_group(required=False)
    neural_parser.add_argument('--neural_augmentation', dest='neural_augmentation', action='store_true')
    neural_parser.add_argument('--no-neural_augmentation', dest='neural_augmentation', action='store_false')
    parser.set_defaults(neural_augmentation=True)
    traditional_parser = parser.add_mutually_exclusive_group(required=False)
    traditional_parser.add_argument('--traditional_augmentation', dest='traditional_augmentation', action='store_true')
    traditional_parser.add_argument('--no-traditional_augmentation', dest='traditional_augmentation', action='store_false')
    parser.set_defaults(traditional_augmentation=True)
    args = parser.parse_args()
    return args


""" Main Program """
if __name__ == '__main__':
    
    ## Utils 
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_duration = 2 
    args = parse_cmd_line_params()


    """ Preprocess Data """
    ## Train
    df_train = pd.read_csv(os.path.join(args.csv_folder, 'train.csv'))
    emotions = df_train['label'].unique()

    ## Prepare the labels
    label2id, id2label = dict(), dict()
    for i, label in enumerate(emotions):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    for index in tqdm(df_train.index):
        # df_train.loc[index,'filename'] = path + df_train.loc[index,'filename']
        df_train.loc[index,'label'] = label2id[df_train.loc[index,'label']]
    df_train['label'] = df_train['label'].astype(int)

    df_valid = pd.read_csv(os.path.join(args.csv_folder, 'devel.csv'))
    for index in tqdm(df_valid.index):
        # df_valid.loc[index,'filename'] = path + df_valid.loc[index,'filename']
        df_valid.loc[index,'label'] = label2id[df_valid.loc[index,'label']]
    df_valid['label'] = df_valid['label'].astype(int)


    """ Define Model """
    if args.feature_extractor == 'wav2vec2':
        model_checkpoint = "facebook/wav2vec2-base"
    elif args.feature_extractor == 'wavlm':
        model_checkpoint = "microsoft/wavlm-base"  
    else:
        print('Feature Extractor Checkpoint not valid! Try again!')
        exit()

    if args.pretrained_model == 'wav2vec2-base':
        model = AutoModelForAudioClassification.from_pretrained(
            'facebook/' + args.pretrained_model, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True)
    elif args.pretrained_model == 'wavlm-base':
        model = AutoModelForAudioClassification.from_pretrained(
            'microsoft/' + args.pretrained_model, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True)
    else:
        model = torch.load(args.pretrained_model) 
        model.num_labels = num_labels
        model.label2id=label2id,
        model.id2label=id2label

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)


    """ Build Dataset """
    train_dataset = finetuning_dataset(df_train, args.wav_folder, feature_extractor, max_duration, device, neural_augmentation=args.neural_augmentation, traditional_augmentation=args.traditional_augmentation)
    valid_dataset = finetuning_dataset(df_valid, args.wav_folder, feature_extractor, max_duration, device, neural_augmentation=args.neural_augmentation, traditional_augmentation=args.traditional_augmentation)

    """ Training Model """
    model_name = model_checkpoint.split("/")[-1]
    batch_size = args.batch_size

    # Define args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.n_epochs,
        warmup_ratio=0.1,
        logging_steps=int(len(train_dataset)/batch_size)+1,
        eval_steps=int(len(valid_dataset)/batch_size)+1,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="uar",
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=args.n_workers,
        dataloader_pin_memory=True,
    )

    ## Class Weights
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df_train["label"]),
        y=np.array(df_train["label"])
    )
    class_weights = torch.tensor(class_weights, device="cuda", dtype=torch.float32)


    ## Trainer 
    #early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    #    callbacks=[early_stopping],
        compute_metrics=compute_metrics
    )

    # Train and Evaluate
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.evaluate()
