import pandas as pd
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
from torch.nn import CrossEntropyLoss
import evaluate
import torchaudio
import transformers
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import json
import click
import numpy as np
from datetime import datetime as dt
from pathlib import Path
from .networks import Wav2Vec2ForSequenceClassificationAux, MTLCrossEntropy, \
    Wav2Vec2ForSequenceClassificationCLSAux, MultiClassMTLCrossEntropy, Wav2Vec2ForSequenceClassificationAuxDeep, \
    MultiClassMTLFocalLoss, FocalLoss
from torch.nn import BCEWithLogitsLoss
from .networks import Wav2Vec2CLSpooling, cls_pooling_mechs
from .dataloading import load_labels, get_combined_train_data, LabelLoader, CompareLabelLoader
import logging
import torch
from datasets import DatasetDict, load_metric, Dataset
from transformers import EarlyStoppingCallback
from transformers import Wav2Vec2ForSequenceClassification, Trainer, \
    Wav2Vec2FeatureExtractor, Wav2Vec2Config, TrainingArguments
from scipy.io import wavfile
from .dataloading import TRAIN_SPK, TEST_SPK, VAL_SPK

CLI_NAME = "mlstutterdetection"


def setup_logging(log_level):
    ext_logger = logging.getLogger(f"py.{CLI_NAME}")
    logging.captureWarnings(True)
    level = getattr(logging, log_level)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=level)
    if level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


@click.group()
@click.option("-l", "--log-level", default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
def mlstutterdetection(log_level):
    setup_logging(log_level)


@mlstutterdetection.command()
@click.option('--sep28k-labels', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips.csv')
@click.option('--sep28k-episodes', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv')
@click.option('--fluencybank-labels', default='SEP-28k/ml-stuttering-events-dataset/fluencybank_debug.csv')
@click.option('--audio-dir', default='SEP-28k/ml-stuttering-events-dataset')
@click.option('--only-clean-labels', is_flag=True, help='use only non-ambiguously labeled clips for training')
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--label-col', default='any', help='label col for the run')
@click.option('--aux-col', default='gender', help='auxiliary target for MTL learning')
@click.option('--epochs', default=10, help='number of epochs to train')
@click.option('--early-stopping', default=3, help='number of epochs allowed for eval loss to worsen')
@click.option('--freeze-encoder', is_flag=True, help='freeze convolutional feature extractor')
@click.option('--tune-layer', is_flag=True, help='tune layers')
@click.option('--w2v2-extract-layer', default=None, help='list of layers/single layer to use for tuning')
@click.option('--gradient-cum-steps', default=5,
              help='Will influence the actual batch size, will be this times batch size')
@click.option('--batch-size', default=32,
              help='Will influence the actual batch size, will be this times gradient cum steps')
@click.option("--stat-pooling", is_flag=True, help='use stats pooling instead of mean pooling')
@click.option('--resume-from', default=None, help='checkpoint to resume training from')
@click.option("--log-dir", default='/tmp')
def fine_tune_w2v2_classifier(sep28k_labels, sep28k_episodes, fluencybank_labels, audio_dir, only_clean_labels, stl,
                              label_col, aux_col, epochs, early_stopping, tune_layer, w2v2_extract_layer,
                              freeze_encoder, gradient_cum_steps, batch_size, stat_pooling, resume_from, log_dir):
    ## detecting dysfluencies in stuttering therapy using wav2vec 2.0
    logger = logging.getLogger(f'py.{CLI_NAME}')
    transformers.logging.set_verbosity_info()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', padding=True)
    logger.info(feature_extractor)
    train = load_labels(sep28k_labels, 'sep28k', audio_dir=audio_dir, extended_episodes_file=sep28k_episodes)
    # 'decision gender wise , August 4th 2022, use host gender if unclear, drop rest
    train.loc[train['gender'] == 'd', 'gender'] = train.loc[train['gender'] == 'd', 'host_gender']
    train.drop(train.loc[train['gender'] == 'd'].index, axis=0, inplace=True)
    val = load_labels(fluencybank_labels, 'fluencybank', audio_dir=audio_dir)

    datasets = DatasetDict(
        {
            'train': Dataset.from_pandas(train[train['clean_label']] if only_clean_labels else train),
            'validation': Dataset.from_pandas(val[val['clean_label']] if only_clean_labels else val)
        })

    # classifier projection size
    # use_weighted_layer_sum
    # consider label smoothing -> not so confident model, helps to tune a better decision boundary
    if tune_layer:
        hidden_layers = range(1, int(w2v2_extract_layer) + 1) if len(w2v2_extract_layer.split(',')) == 1 else [int(l) for l in w2v2_extract_layer.split(',')]
    else:
        hidden_layers = [int(w2v2_extract_layer) if w2v2_extract_layer is not None else 12]

    for num_hidden_layers in hidden_layers:
        now = dt.now()
        time_stamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        training_name = f'{time_stamp}_{label_col}_{aux_col}_{num_hidden_layers}_sep28k_finetuned' if not stl else f'{time_stamp}_{label_col}_{num_hidden_layers}_sep28k_finetuned_stl'
        exp_log_dir = f'{log_dir}/{training_name}'
        logger.info(f'{50 * "*"} training {exp_log_dir} starting {50 * "*"}')
        args = TrainingArguments(
            exp_log_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,  # 32 is the max the gpus can take
            gradient_accumulation_steps=gradient_cum_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            push_to_hub=False,
            logging_dir=exp_log_dir
        )

        aux_labels = pd.Series(datasets['train'][aux_col]).value_counts()

        label2id, id2label = {}, {}
        for i, label in enumerate(aux_labels.items()):
            label2id[label[0]] = i
            id2label[i] = label[0]

        def prepare_datasets(example):
            _, wav = wavfile.read(example['path'])
            example['audio'] = norm_wav(wav)
            example['label'] = int(example[label_col])
            if not stl:
                example['aux_labels'] = label2id.get(example[aux_col], 0)
            return example

        loaded_wavs = datasets.map(prepare_datasets)

        def preprocess_data(examples):
            audios = [x for x in examples['audio']]

            inputs = feature_extractor(
                audios,
                sampling_rate=16000,
                max_length=int(3.0 * 16000),
                truncation=True,
                padding=True
            )
            return inputs

        encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)
        class_weights = torch.Tensor(
            1 - pd.Series(encoded_datasets['train'][label_col]).value_counts(normalize=True).sort_index())
        logger.info(f'class weights: {class_weights}')

        if stl:
            aux_weights, loss = None, None
            num_aux_labels = 2
        else:
            aux_weights = torch.Tensor(
                1 - pd.Series(encoded_datasets['train']['aux_labels']).value_counts(normalize=True).sort_index())
            loss = MTLCrossEntropy(main_loss_weight=0.9, class_weights=class_weights,
                                   aux_class_weights=aux_weights, num_labels=len(class_weights),
                                   num_aux_labels=len(aux_weights))
            num_aux_labels = len(aux_weights)
            logger.info(f'aux class weights: {aux_weights}')

        seq_clf = Wav2Vec2ForSequenceClassificationAux(Wav2Vec2Config()).from_pretrained(
            'facebook/wav2vec2-base-960h', config=Wav2Vec2Config(num_labels=2, num_hidden_layers=num_hidden_layers),
            num_aux_labels=num_aux_labels,
            stat_pooling=stat_pooling, loss=loss)

        if freeze_encoder:
            seq_clf.freeze_feature_encoder()

        trainer = Trainer(
            seq_clf,
            args,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["validation"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
        )

        trainer.train() if resume_from is None else trainer.train(resume_from)
        val_metrics = trainer.evaluate()
        train_metrics = trainer.evaluate(encoded_datasets['train'], metric_key_prefix='train')
        trainer.log_metrics('train', train_metrics)
        trainer.log_metrics('eval', val_metrics)
        trainer.save_metrics(split='all', metrics={**val_metrics, **train_metrics, 'layer': num_hidden_layers})
        trainer.save_model(exp_log_dir)
        logger.info(f'{50 * "*"}training {exp_log_dir} done {50 * "*"}')


def get_best_weights_from_transformer_dir(path):
    path = Path(path)
    dirs = sorted(p for p in path.iterdir() if 'checkpoint' in p.name)
    is_checkpoint_dir = (dirs[0] / 'trainer_state.json').exists()
    if is_checkpoint_dir:
        with open(dirs[0] / 'trainer_state.json') as f:
            info = json.load(f)
        return path / info['best_model_checkpoint'].split('/')[-1]
    else:
        return str(path)


@mlstutterdetection.command()
@click.option('--ksof-labels', required=True)
@click.option('--audio-dir', required=True)
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--label-col', default='any', help='label col for the run')
@click.option('--aux-col', default='gender', help='auxiliary target for MTL learning')
@click.option('--epochs', default=10, help='number of epochs to train')
@click.option('--early-stopping', default=3, help='number of epochs allowed for eval loss to worsen')
@click.option('--weights', default=None, help='model training dir, with trainer_state.json, loads best checkpoint')
@click.option('--freeze-w2v2', is_flag=True, help='freezes the Wav2Vec2 module')
@click.option('--use-weighted-layer-sum', is_flag=True, help='as it said')
@click.option('--w2v2-extract-layer', default=None,
              help='only works if freeze is true, extracts features after the w2v2 transformer block at layer(1-12)')
@click.option('--tune-layer', is_flag=True, help='tunes hum_hidden_layers for i in range(1, w2v2_extract_layer')
@click.option('--gradient-cum-steps', default=5, help='Will influence the actual batch size, will be this times 32')
@click.option('--batch-size', default=32, help='number of samples per batch (actual, times gradient-cum-steps)')
@click.option("--stat-pooling", is_flag=True, help='use stats pooling instead of mean pooling')
@click.option("--log-dir", default='/tmp')
def train_ksof_w2v2_classifier(ksof_labels, audio_dir, stl, label_col, aux_col, epochs, early_stopping,
                               weights, freeze_w2v2, use_weighted_layer_sum, w2v2_extract_layer, tune_layer,
                               batch_size, gradient_cum_steps, stat_pooling, log_dir):
    ## Detecting dysfluencies in stuttering therapy using wav2vec 2.0
    # 'gender', 'clean' / 'dirty' flag, CCC loss
    # classification head with attention, classification head with avg_pooling with stride?
    logger = logging.getLogger(f'py.{CLI_NAME}')

    label_df = load_labels(ksof_labels, 'ksof', audio_dir=audio_dir)

    datasets = DatasetDict(
        {'train': Dataset.from_pandas(label_df[label_df['speaker'].isin(TRAIN_SPK)]),
         'test': Dataset.from_pandas(label_df[label_df['speaker'].isin(TEST_SPK)]),
         'validation': Dataset.from_pandas(label_df[label_df['speaker'].isin(VAL_SPK)])
         })
    # classifier projection size
    # use_weighted_layer_sum
    # consider label smoothing -> not so confident model, helps to tune a better decision boundary
    aux_labels = pd.Series(datasets['train'][aux_col]).value_counts()

    label2id, id2label = {}, {}
    for i, label in enumerate(aux_labels.items()):
        label2id[label[0]] = i
        id2label[i] = label[0]

    def prepare_datasets(example):
        _, wav = wavfile.read(example['path'])
        example['audio'] = norm_wav(wav)
        example['label'] = int(example[label_col])
        if not stl:
            example['aux_labels'] = label2id.get(example[aux_col], 0)
        return example

    loaded_wavs = datasets.map(prepare_datasets)

    if tune_layer:
        hidden_layers = range(1, int(w2v2_extract_layer) + 1) if len(w2v2_extract_layer.split(',')) == 1 else [int(l) for l in w2v2_extract_layer.split(',')]
    else:
        hidden_layers = [int(w2v2_extract_layer) if w2v2_extract_layer is not None and freeze_w2v2 else 12]

    for num_hidden_layers in hidden_layers:

        now = dt.now()
        time_stamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        training_name = f'{time_stamp}_{label_col}_{aux_col}_{num_hidden_layers}_ksof_finetuned' if not stl else f'{time_stamp}_{label_col}_{num_hidden_layers}_ksof_finetuned_stl'
        exp_log_dir = f'{log_dir}/{label_col}/{training_name}'

        args = TrainingArguments(
            output_dir=exp_log_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,  # 32 is the max the gpus can take
            gradient_accumulation_steps=gradient_cum_steps,
            per_device_eval_batch_size=batch_size,
            eval_accumulation_steps=2,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            push_to_hub=False,
            logging_dir=exp_log_dir
        )
        model_str = get_best_weights_from_transformer_dir(
            weights) if weights is not None else 'facebook/wav2vec2-base-960h'
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_str, padding=True)
        logger.info(feature_extractor)

        def preprocess_data(examples):
            audios = [x for x in examples['audio']]
            inputs = feature_extractor(
                audios,
                sampling_rate=16000,
                max_length=int(3.0 * 16000),
                truncation=True,
                padding=True
            )
            return inputs

        encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)
        class_weights = torch.Tensor(
            1 - pd.Series(encoded_datasets['train'][label_col]).value_counts(normalize=True).sort_index())
        logger.info(f'class weights: {class_weights}')

        if stl:
            aux_weights, loss = None, None
            num_aux_labels = 2
        else:
            aux_weights = torch.Tensor(
                1 - pd.Series(encoded_datasets['train']['aux_labels']).value_counts(normalize=True).sort_index())
            loss = MTLCrossEntropy(main_loss_weight=0.9, class_weights=class_weights,
                                   aux_class_weights=aux_weights, num_labels=len(class_weights),
                                   num_aux_labels=len(aux_weights))
            num_aux_labels = len(aux_weights)
            logger.info(f'aux class weights: {aux_weights}')

        seq_clf = Wav2Vec2ForSequenceClassificationAux.from_pretrained(
            model_str, config=Wav2Vec2Config(num_labels=2, num_hidden_layers=num_hidden_layers,
                                             use_weighted_layer_sum=use_weighted_layer_sum),
            num_aux_labels=num_aux_labels,
            stat_pooling=stat_pooling, loss=loss)

        if freeze_w2v2:
            seq_clf.wav2vec2.requires_grad_(False)

        logger.info(seq_clf)
        logger.info(f'Model parameters {sum(p.numel() for p in seq_clf.parameters())}')
        logger.info(f'Trainable model parameters {sum(p.numel() for p in seq_clf.parameters() if p.requires_grad)}')

        trainer = Trainer(
            seq_clf,
            args,
            train_dataset=encoded_datasets['train'],
            eval_dataset=encoded_datasets['validation'],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
        )

        trainer.train()
        val_metrics = trainer.evaluate()
        test_metrics = trainer.evaluate(encoded_datasets['test'], metric_key_prefix='test')
        trainer.log_metrics('eval', val_metrics)
        trainer.log_metrics('test', test_metrics)
        trainer.save_metrics(split='all', metrics={**val_metrics, **test_metrics, 'layer': num_hidden_layers})
        trainer.save_model(exp_log_dir)


@mlstutterdetection.command()
@click.option('--ksof-labels', required=True)
@click.option('--audio-dir', required=True)
@click.option('--label-col', default='any', help='label col for true labels')
@click.option('--weights', required=True, help='path to model weights')
@click.option('--results-out', default=None, help='path to csv output for results')
def make_prediction_ksof_w2v2(ksof_labels, audio_dir, label_col, weights, results_out):
    # best mixed model: /Users/bayerl/projects/w2v2_stutter_demo/its/ressources/2022_9_29_11_20_4_multiclass_any_0.99_12_mixed_finetuned
    # 'gender', 'clean' / 'dirty' flag, CCC loss
    # classification head with attention, classification head with avg_pooling with stride?
    logger = logging.getLogger(f'py.{CLI_NAME}')
    from transformers import pipeline
    # logger.info(feature_extractor)
    label_df = load_labels(ksof_labels, 'ksof', audio_dir=audio_dir)

    model_str = get_best_weights_from_transformer_dir(weights)
    pipe = pipeline('audio-classification', model=model_str, feature_extractor=model_str)
    datasets = {
        'test': label_df[label_df['speaker'].isin(TEST_SPK)],
        'validation': label_df[label_df['speaker'].isin(VAL_SPK)]
    }

    from tqdm.auto import tqdm
    tmp_results = []
    for split, df in datasets.items():
        predictions = {}
        for row in tqdm(df[['path', label_col]].to_numpy()[0:10]):
            out = pipe(row[0], top_k=1)[0]  # only first list item
            out['actual'] = int(row[1])
            predictions[row[0]] = out
        y_pred = [int(pred['label'].split('_')[1]) for _, pred in predictions.items()]
        y_true = [pred['actual'] for _, pred in predictions.items()]
        f1 = f1_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        precision = precision_score(y_true, y_pred, average=None)
        tmp_results.append({f'{split}_recall': recall,
                            f'{split}_f1': f1,
                            f'{split}_precision': precision,
                            })
    results = {**tmp_results[0], **tmp_results[1], 'name': model_str}
    results = pd.DataFrame(results).T
    if results_out is not None:
        results.to_csv(results_out)
    return results


def compute_metrics(eval_pred):
    """Computes metrics on a batch of predictions"""
    metrics = {k: load_metric(k) for k in ['precision', 'recall', 'f1']}
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    output = list(
        [v.compute(predictions=predictions, references=eval_pred.label_ids, pos_label=1) for v in metrics.values()])
    return {**output[0], **output[1], **output[2]}


def compute_multilabel_metrics(eval_pred):
    """Computes metrics on a batch of predictions"""
    metrics = {k: evaluate.load(k, config_name='multilabel') for k in ['precision', 'recall', 'f1']}
    # predictions = np.round(eval_pred.predictions) round fails, values can be greater than 1, if no sigmoid is applied *which is in BCE
    if isinstance(eval_pred.predictions, np.ndarray):
        predictions = torch.round(
            torch.sigmoid(torch.tensor(eval_pred.predictions)))
    else:
        predictions = torch.round(
            torch.sigmoid(torch.tensor(eval_pred.predictions[0])))  # this doenst work if weighted layer sum

    try:
        if isinstance(eval_pred.label_ids, tuple):
            labels = eval_pred.label_ids[0]  # we only need the main predictions for the metrics
        else:
            labels = eval_pred.label_ids
        output = list(
            [v.compute(predictions=predictions, references=labels, pos_label=1, average=None) for v in
             metrics.values()])
    except ValueError as e:
        print(e)
        print(predictions)
        print(predictions.shape)
        print(eval_pred.label_ids)
        print(eval_pred.label_ids.shape)
        # tmp = {'eval_loss': 0.845672607421875, 'eval_precision_SoundRep': 0.07331480194579569, 'eval_precision_Prolongation': 0.11487146111291897, 'eval_precision_Block': 0.11911911911911911, 'eval_precision_WordRep': 0.11951538965291421, 'eval_precision_Interjection': 0.2246163891609533, 'eval_precision_NoStutteredWords': 0.6147149677057007, 'eval_recall_SoundRep': 0.4806378132118451, 'eval_recall_Prolongation': 0.530827067669173, 'eval_recall_Block': 0.4530456852791878, 'eval_recall_WordRep': 0.5305232558139535, 'eval_recall_Interjection': 0.5375, 'eval_recall_NoStutteredWords': 0.5736373165618449, 'eval_f1_SoundRep': 0.12722339463370513, 'eval_f1_Prolongation': 0.18887105403959337, 'eval_f1_Block': 0.18863936591809774, 'eval_f1_WordRep': 0.19508284339925175, 'eval_f1_Interjection': 0.31683168316831684, 'eval_f1_NoStutteredWords': 0.5934661786634133, 'eval_runtime': 153.4072, 'eval_samples_per_second': 42.775, 'eval_steps_per_second': 0.169, 'epoch': 2.0}
        # output = [list(), list(), list()]
        # output[0] = {'precision': [v for k, v in tmp.items() if 'precision' in k]}
        # output[1] = {'recall': [v for k, v in tmp.items() if 'recall' in k]}
        # output[2] = {'f1': [v for k, v in tmp.items() if 'f1' in k]}
        # output[0] = {'_'.join(k.split('_')[0:1]): v for k, v in tmp.items() if 'f1' in k}

    col_names = LabelLoader().stuttering_cols if len(eval_pred.label_ids[0]) == 6 else LabelLoader().non_fluent_cols
    output[0] = {f'precision_{col_names[k]}': v.item() for k, v in enumerate(output[0]['precision'])}
    output[1] = {f'recall_{col_names[k]}': v.item() for k, v in enumerate(output[1]['recall'])}
    output[2] = {f'f1_{col_names[k]}': v.item() for k, v in enumerate(output[2]['f1'])}
    return {
        **output[0], 'precision_macro': np.mean([v for k, v in output[0].items() if 'NoStutteredWords' not in k]),
        **output[1], 'recall_macro': np.mean([v for k, v in output[1].items() if 'NoStutteredWords' not in k]),
        **output[2], 'f1_macro': np.mean([v for k, v in output[2].items() if 'NoStutteredWords' not in k])
    }


@mlstutterdetection.command()
@click.option('--labels', required=True, help='label csv file, sep28k, ksof, fluencybank')
@click.option('--kind', required=True, default='ksof')
@click.option('--audio-dir', required=True)
@click.option('--model-weights', default=None, help='if none, use wav2vec2-base-960h')
@click.option('--w2v2-extract-layer', default=12, help='check and get from model, as those where tuned by layer')
@click.option('--extract-all', is_flag=True, help='extract all layers up to w2v2-extract-layer')
@click.option("--out-dir", default='/tmp')
@click.option("--nj", default=8)
def extract_feats_from_fine_tuned(labels, kind, audio_dir, model_weights, w2v2_extract_layer, extract_all, out_dir, nj):
    torch.set_num_threads(nj)
    logger = logging.getLogger(f'py.{CLI_NAME}')
    out_dir = Path(out_dir)

    df_labels = load_labels(labels, kind, audio_dir=audio_dir)
    data = Dataset.from_pandas(df_labels)

    model_weights = 'facebook/wav2vec2-base-960h' if model_weights is None else get_best_weights_from_transformer_dir(
        model_weights)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_weights, padding=True)
    logger.info(feature_extractor)
    extractor = Wav2Vec2ForSequenceClassificationAux(Wav2Vec2Config()).from_pretrained(
        model_weights, config=Wav2Vec2Config(num_labels=2, num_hidden_layers=int(w2v2_extract_layer)))

    logger.info(extractor)

    def preprocess_data(examples):
        audios = [x for x in examples['audio']]

        inputs = feature_extractor(
            audios,
            sampling_rate=16000,
            max_length=int(3.0 * 16000),
            truncation=True,
            padding=True
        )
        return inputs

    def prepare_datasets(example):
        _, wav = wavfile.read(example['path'])
        example['audio'] = norm_wav(wav)
        return example

    loaded_wavs = data.map(prepare_datasets)
    encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)

    extractor.freeze_feature_encoder()
    extractor.wav2vec2.requires_grad_(False)

    from tqdm.auto import tqdm
    features = []

    for i, datum in tqdm(enumerate(encoded_datasets)):
        # assert(df_labels.iloc[i]['segment_name'] == datum['segment_name'])
        feats = extractor(torch.tensor(datum['input_values']).reshape(1, -1), output_hidden_states=True)
        if extract_all:
            # inclusive / exclusive -> +1, 0 holds feats after convolutional part
            features.append(feats['hidden_states'][1:int(w2v2_extract_layer) + 1])
        else:
            features.append(feats['hidden_states'][int(w2v2_extract_layer)].detach())

    out_file = f'{kind}_1-{w2v2_extract_layer}_{Path(model_weights).name}.pkl' if extract_all else f'{kind}_{w2v2_extract_layer}_{Path(model_weights).name}.pkl' # TODO change from pkl to apache arrow/ parquet

    torch.save((df_labels, features), out_dir / out_file)


@mlstutterdetection.command()
@click.option('--sep28k-labels', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips_debug.csv')
@click.option('--sep28k-episodes', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv')
@click.option('--fluencybank-labels', default='SEP-28k/ml-stuttering-events-dataset/fluencybank_debug.csv')
@click.option('--audio-dir', default='SEP-28k/ml-stuttering-events-dataset')
@click.option('--ksof-labels', default='KST/segments/ksof_debug.csv')
@click.option('--ksof-audio-dir', default='KST/')
@click.option('--only-clean-labels', is_flag=True, help='use only non-ambiguously labeled clips for training')
@click.option("--pooling-mechanism", type=click.Choice(cls_pooling_mechs.keys()), default='projection',
              help='which pooling mech to use, matrix only works for fixed size inputs')
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--label-col', default='any', help='label col for the run')
@click.option('--aux-col', default='gender', help='auxiliary target for MTL learning')
@click.option('--epochs', default=10, help='number of epochs to train')
@click.option('--early-stopping', default=3, help='number of epochs allowed for eval loss to worsen')
@click.option('--freeze-encoder', is_flag=True, help='freeze convolutional feature extractor')
@click.option('--tune-layer', is_flag=True, help='tune layers')
@click.option('--w2v2-extract-layer', default=None, help='list of layers/single layer to use for tuning')
@click.option('--gradient-cum-steps', default=5,
              help='Will influence the actual batch size, will be this times batch size')
@click.option('--batch-size', default=32,
              help='Will influence the actual batch size, will be this times gradient cum steps')
@click.option('--resume-from', default=None, help='checkpoint to resume training from')
@click.option("--log-dir", default='/tmp')
def fine_tune_w2v2_cls_pooling(sep28k_labels, sep28k_episodes, fluencybank_labels, audio_dir,
                               ksof_labels, ksof_audio_dir, only_clean_labels,
                               stl, pooling_mechanism, label_col, aux_col, epochs,
                               early_stopping, tune_layer, w2v2_extract_layer,
                               freeze_encoder, gradient_cum_steps, batch_size, resume_from, log_dir):
    # different heads possible. in essence 
    # A Stutter Seldom Comes Alone – Cross-Corpus Stuttering Detection as a Multi-Label Problem
    logger = logging.getLogger(f'py.{CLI_NAME}')
    transformers.logging.set_verbosity_info()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', padding=True)
    logger.info(feature_extractor)

    # 'clean_label'
    train = load_labels(sep28k_labels, 'sep28k', audio_dir=audio_dir, extended_episodes_file=sep28k_episodes)
    # 'decision gender wise , August 4th 2022, use host gender if unclear, drop rest
    train.loc[train['gender'] == 'd', 'gender'] = train.loc[train['gender'] == 'd', 'host_gender']
    train.drop(train.loc[train['gender'] == 'd'].index, axis=0, inplace=True)
    fbank = load_labels(fluencybank_labels, 'fluencybank', audio_dir=audio_dir)
    ksof = load_labels(ksof_labels, 'ksof', audio_dir=ksof_audio_dir)
    datasets = DatasetDict(
        {
            'train': Dataset.from_pandas(train[train['clean_label']] if only_clean_labels else train),
            'validation': Dataset.from_pandas(fbank[fbank['clean_label']] if only_clean_labels else fbank),
            'test': Dataset.from_pandas(ksof[ksof['clean_label']] if only_clean_labels else ksof)

        })

    # classifier projection size
    # use_weighted_layer_sum
    # consider label smoothing -> not so confident model, helps to tune a better decision boundary
    if tune_layer:
        hidden_layers = range(1, int(w2v2_extract_layer) + 1) if len(w2v2_extract_layer.split(',')) == 1 else [int(l) for l in w2v2_extract_layer.split(',')]
    else:
        # just one layer, list format for easy iteration
        hidden_layers = [int(w2v2_extract_layer) if w2v2_extract_layer is not None else 12]

    for num_hidden_layers in hidden_layers:
        now = dt.now()
        time_stamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        training_name = f'{time_stamp}_{label_col}_{aux_col}_{num_hidden_layers}_sep28k_finetuned_cls_{pooling_mechanism}' if not stl else f'{time_stamp}_{label_col}_{num_hidden_layers}_sep28k_finetuned_stl_cls_{pooling_mechanism}'
        exp_log_dir = f'{log_dir}/{training_name}'
        logger.info(f'{50 * "*"} training {exp_log_dir} starting {50 * "*"}')
        args = TrainingArguments(
            exp_log_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,  # 32 is the max the gpus can take
            gradient_accumulation_steps=gradient_cum_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            push_to_hub=False,
            logging_dir=exp_log_dir
        )

        aux_labels = pd.Series(datasets['train'][aux_col]).value_counts()

        label2id, id2label = {}, {}
        for i, label in enumerate(aux_labels.items()):
            label2id[label[0]] = i
            id2label[i] = label[0]

        def prepare_datasets(example):
            _, wav = wavfile.read(example['path'])
            example['audio'] = norm_wav(wav)
            example['label'] = int(example[label_col])
            if not stl:
                example['aux_labels'] = label2id.get(example[aux_col], 0)
            return example

        loaded_wavs = datasets.map(prepare_datasets)

        def preprocess_data(examples):
            audios = [x for x in examples['audio']]

            inputs = feature_extractor(
                audios,
                sampling_rate=16000,
                max_length=int(3.0 * 16000),
                truncation=True,
                padding=True
            )
            return inputs

        encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)
        class_weights = torch.Tensor(
            1 - pd.Series(encoded_datasets['train'][label_col]).value_counts(normalize=True).sort_index())
        logger.info(f'class weights: {class_weights}')

        if stl:
            aux_weights, loss = None, None
            num_aux_labels = 2
        else:
            aux_weights = torch.Tensor(
                1 - pd.Series(encoded_datasets['train']['aux_labels']).value_counts(normalize=True).sort_index())
            loss = MTLCrossEntropy(main_loss_weight=0.9, class_weights=class_weights,
                                   aux_class_weights=aux_weights, num_labels=len(class_weights),
                                   num_aux_labels=len(aux_weights))
            num_aux_labels = len(aux_weights)
            logger.info(f'aux class weights: {aux_weights}')

        seq_clf = Wav2Vec2ForSequenceClassificationCLSAux(Wav2Vec2Config()).from_pretrained(
            'facebook/wav2vec2-base-960h', config=Wav2Vec2Config(num_labels=2, num_hidden_layers=num_hidden_layers),
            num_aux_labels=num_aux_labels,
            loss=loss, pooling_mechanism=pooling_mechanism)

        if freeze_encoder:
            seq_clf.freeze_feature_encoder()

        trainer = Trainer(
            seq_clf,
            args,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["validation"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
        )

        trainer.train() if resume_from is None else trainer.train(resume_from)
        val_metrics = trainer.evaluate()
        test_metrics = trainer.evaluate(encoded_datasets['test'])
        train_metrics = trainer.evaluate(encoded_datasets['train'], metric_key_prefix='train')
        trainer.log_metrics('train', train_metrics)
        trainer.log_metrics('eval', val_metrics)
        trainer.log_metrics('test', test_metrics)
        trainer.save_metrics(split='all', metrics={**val_metrics, **train_metrics, 'layer': num_hidden_layers})
        trainer.save_model(exp_log_dir)
        logger.info(f'{50 * "*"} training {exp_log_dir} done {50 * "*"}')


@mlstutterdetection.command()
def debug():
    wav, rate = torchaudio.load('/Users/bayerl/projects/sep28-reproduce/SEP-28k/ml-stuttering-events-dataset/clips/FluencyBank/016/FluencyBank_016_0.wav')
    seq_clf = Wav2Vec2ForSequenceClassificationCLSAux(Wav2Vec2Config()).from_pretrained(
    'facebook/wav2vec2-base-960h', config=Wav2Vec2Config(num_labels=2, num_hidden_layers=12),
     pooling_mechanism='projection')
    print(seq_clf)
    result = seq_clf(wav)
    print(result)


@mlstutterdetection.command()
@click.option('--sep28k-labels', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips.csv')
@click.option('--sep28k-episodes', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv')
@click.option('--fluencybank-labels', default='SEP-28k/ml-stuttering-events-dataset/fluencybank_debug.csv')
@click.option('--audio-dir', default='SEP-28k/ml-stuttering-events-dataset')
@click.option('--ksof-labels', help='ksof label file, need if train data is one of ksof/mix')
@click.option('--ksof-audio-dir',
              help='dir containing the KST dataset with clips, needed when train data is one of ksof/mix')
@click.option('--train-data', help='name of train data to use',
              type=click.Choice(['ksof', 'sep28k', 'fbank', 'sep28fbankfulldev', 'eng', 'smallmixed', 'mixed']),
              default='sep28k')
@click.option("--clf-type", type=click.Choice(['mean', 'deepmean', 'attention', 'sequencetoken']),
              help='which clf version to use')
@click.option("--clf-dropout", is_flag=True, help='use dropout in clf')
@click.option('--model', default='facebook/wav2vec2-base-960h', help='path to model/ model name for huggingface')
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--focal-loss', is_flag=True, help='use focal loss instead of vanilla CRE')
@click.option('--focal-loss-alpha', default=None, help='focal loss alpha parameter')
@click.option('--focal-loss-gamma', default=2, help='focal loss gamma parameter')
@click.option('--focal-loss-reduction', default='mean', type=click.Choice(['mean', 'sum']))
@click.option('--aux-col', default='gender', help='auxiliary target for MTL learning')
@click.option('--epochs', default=10, help='number of epochs to train')
@click.option('--early-stopping', default=3, help='number of epochs allowed for eval loss to worsen')
@click.option('--main-loss-weight', default=0.9,
              help='weight of the dysfluency loss, aux_loss_weight = 1 - main_main_loss_weight ')
@click.option('--num-hidden-layers', default=12, help='number of transformer layers')
@click.option('--use-weighted-layer-sum', is_flag=True, help='what it says')
@click.option('--freeze-encoder', is_flag=True, help='freeze convolutional feature extractor')
@click.option('--freeze-wav2vec', is_flag=True, help='freeze wav2vec portion, only train clf head')
@click.option('--gradient-cum-steps', default=5,
              help='Will influence the actual batch size, will be this times batch size')
@click.option('--batch-size', default=32,
              help='Will influence the actual batch size, will be this times gradient cum steps')
@click.option('--eval-accumulation-steps', default=None, help='avoids CUDA OOM error in eval, slows down eval')
@click.option("--stat-pooling", is_flag=True, help='use stats pooling instead of mean pooling')
@click.option('--resume-from', default=None, help='checkpoint to resume training from')
@click.option("--log-dir", default='/tmp')
def fine_tune_multilabel_w2v2_classifier(sep28k_labels, sep28k_episodes, fluencybank_labels, audio_dir,
                                         ksof_labels, ksof_audio_dir, train_data, clf_type, clf_dropout, model,
                                         stl, focal_loss, focal_loss_alpha, focal_loss_gamma, focal_loss_reduction,
                                         aux_col, epochs, early_stopping, main_loss_weight,
                                         num_hidden_layers, use_weighted_layer_sum,
                                         freeze_encoder, freeze_wav2vec, gradient_cum_steps,
                                         batch_size, eval_accumulation_steps, stat_pooling,
                                         resume_from, log_dir):
    # Meat of: A Stutter Seldom Comes Alone – Cross-Corpus Stuttering Detection as a Multi-Label Problem
    logger = logging.getLogger(f'py.{CLI_NAME}')
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    transformers.logging.set_verbosity_info()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model, padding=True)
    logger.info(feature_extractor)
    train, dev, test = get_combined_train_data(kind=train_data, sep28k_labels=sep28k_labels,
                                               sep28k_episodes=sep28k_episodes,
                                               fbank_labels=fluencybank_labels, ksof_labels=ksof_labels,
                                               sep_audio_dir=audio_dir, ksof_audio_dir=ksof_audio_dir)
    # train = load_labels(sep28k_labels, 'sep28k', audio_dir=audio_dir, extended_episodes_file=sep28k_episodes)
    for ds in [train, test, dev]:
        try:
            # if fluencybank_val:
            # 'decision gender wise , August 4th 2022, use host gender if unclear, drop rest
            ds.loc[train['gender'] == 'd', 'gender'] = ds.loc[train['gender'] == 'd', 'host_gender']
        except Exception as e:
            logging.info(f'was not sep dataset: {e}')
            pass
        ds.drop(train.loc[train['gender'] == 'd'].index, axis=0, inplace=True)

    label_cols = [f'is_{col}' for col in LabelLoader().stuttering_cols]
    label_cols = [*label_cols, 'is_Modified'] if train_data in ['ksof', 'mixed', 'smallmixed'] else label_cols
    logger.info(f'{10 * "*"} training with num_labels: {len(label_cols)} {10 * "*"}')

    datasets = DatasetDict(
        {
            'train': Dataset.from_pandas(train),
            'validation': Dataset.from_pandas(dev),
            'test': Dataset.from_pandas(test),
        })
    # classifier projection size
    # use_weighted_layer_sum
    # consider label smoothing -> not so confident model, helps to tune a better decision boundary

    now = dt.now()
    time_stamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    training_name = f'{time_stamp}_multiclass_{aux_col}_{main_loss_weight}_{num_hidden_layers}_{train_data}_finetuned' if not stl else f'{time_stamp}_multiclass_{num_hidden_layers}_{train_data}_finetuned_stl'
    exp_log_dir = f'{log_dir}/{training_name}'
    logger.info(f'{40 * "*"} training {exp_log_dir} starting {40 * "*"}')
    args = TrainingArguments(
        exp_log_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,  # 32 is the max the gpus can take
        gradient_accumulation_steps=gradient_cum_steps,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=int(
            eval_accumulation_steps) if eval_accumulation_steps is not None else eval_accumulation_steps,
        # necessary, otherwise oom on CUDA during eval
        num_train_epochs=int(epochs),
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",   # f1_macro?
        push_to_hub=False,
        logging_dir=exp_log_dir
    )

    aux_labels = pd.Series(datasets['train'][aux_col]).value_counts()

    label2id, id2label = {}, {}
    for i, label in enumerate(aux_labels.items()):
        label2id[label[0]] = i
        id2label[i] = label[0]

    def prepare_datasets(example):
        _, wav = wavfile.read(example['path'])
        example['audio'] = norm_wav(wav)
        example['label'] = torch.Tensor([example[col] for col in label_cols])
        if not stl:
            example['aux_labels'] = label2id.get(example[aux_col], 0)
        return example

    loaded_wavs = datasets.map(prepare_datasets)

    def preprocess_data(examples):
        audios = [x for x in examples['audio']]

        inputs = feature_extractor(
            audios,
            sampling_rate=16000,
            max_length=int(3.0 * 16000),
            truncation=True,
            padding=True
        )
        return inputs

    encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)

    dist = torch.Tensor(train[label_cols].sum()[label_cols] / len(train))
    class_weights = 1 - dist
    pos_weight = class_weights / dist
    logger.info(f'class weights: {class_weights}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    focal_loss_alpha = float(focal_loss_alpha) if focal_loss_alpha is not None else torch.tensor(class_weights,
                                                                                                 device=device)
    if stl:
        aux_weights = None
        if focal_loss:
            loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction=focal_loss_reduction)
        else:
            loss = BCEWithLogitsLoss(reduction='mean', weight=class_weights, pos_weight=pos_weight)
        num_aux_labels = 2
    else:
        aux_weights = 1 - torch.Tensor(
            pd.Series(encoded_datasets['train']['aux_labels']).value_counts(normalize=True).sort_index())

        if focal_loss:
            loss = MultiClassMTLFocalLoss(main_loss_weight=main_loss_weight, alpha=focal_loss_alpha,
                                          gamma=focal_loss_gamma, reduction=focal_loss_reduction,       #   TODO rerun and double check before camera ready ICASSP!
                                          aux_class_weights=aux_weights, num_aux_labels=len(aux_weights))
        else:
            loss = MultiClassMTLCrossEntropy(main_loss_weight=main_loss_weight, class_weights=class_weights,
                                             aux_class_weights=aux_weights, num_labels=len(class_weights),
                                             num_aux_labels=len(aux_weights), pos_weight=pos_weight)
        num_aux_labels = len(aux_weights)
        logger.info(f'aux class weights: {aux_weights}')

    config = Wav2Vec2Config.from_pretrained(model)
    config.num_labels = len(label_cols)
    config.num_hidden_layers = num_hidden_layers
    config.use_weighted_layer_sum = use_weighted_layer_sum
    if clf_type == 'deepmean':
        config.classifier_proj_size = 512
        seq_clf = Wav2Vec2ForSequenceClassificationAuxDeep.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config,
            num_aux_labels=num_aux_labels,
            stat_pooling=stat_pooling, loss=loss)
    elif clf_type == 'attention':
        seq_clf = Wav2Vec2ForSequenceClassificationCLSAux.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config,
            num_aux_labels=num_aux_labels,
            loss=loss, pooling_mechanism='attention')
        # deprecated:
        # seq_clf = Wav2Vec2ForSequenceClassificationAuxAttentionHead(Wav2Vec2Config()).from_pretrained(
        #     'facebook/wav2vec2-base-960h',
        #     config=Wav2Vec2Config(num_labels=len(label_cols), num_hidden_layers=num_hidden_layers,
        #                           use_weighted_layer_sum=use_weighted_layer_sum),
        #     num_aux_labels=num_aux_labels,
        #     stat_pooling=stat_pooling, loss=loss)
    elif clf_type == 'sequencetoken':
        seq_clf = Wav2Vec2ForSequenceClassificationCLSAux.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config,
            num_aux_labels=num_aux_labels,
            loss=loss, pooling_mechanism='token')
    else:
        seq_clf = Wav2Vec2ForSequenceClassificationAux.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config,
            num_aux_labels=num_aux_labels,
            stat_pooling=stat_pooling, loss=loss)

    if freeze_encoder:
        seq_clf.freeze_feature_encoder()

    if freeze_wav2vec:
        seq_clf.freeze_feature_encoder()
        if hasattr(seq_clf, 'wav2vec'):
            for param in seq_clf.wav2vec.parameters():
                param.requires_grad = False
        else:
            for param in seq_clf.wav2vec2.parameters():
                param.requires_grad = False

    trainer = Trainer(
        seq_clf,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_multilabel_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
    )

    trainer.train() if resume_from is None else trainer.train(resume_from)
    # too large when using weighted layer sum
    # train_metrics = trainer.evaluate(encoded_datasets['train'], metric_key_prefix='train')
    # trainer.log_metrics('train', train_metrics)
    val_metrics = trainer.evaluate()
    trainer.log_metrics('eval', val_metrics)
    if train_data not in ['ksof', 'sep28k', 'fbank']:
        sep_test, fbank_test, ksof_test = get_combined_train_data(kind=train_data, sep28k_labels=sep28k_labels,
                                                                  sep28k_episodes=sep28k_episodes,
                                                                  fbank_labels=fluencybank_labels,
                                                                  ksof_labels=ksof_labels,
                                                                  sep_audio_dir=audio_dir,
                                                                  ksof_audio_dir=ksof_audio_dir,
                                                                  get_test_data=True)
        test_datasets = DatasetDict(
            {
                'sep_test': Dataset.from_pandas(sep_test),
                'fbank_test': Dataset.from_pandas(fbank_test),
                'ksof_test': Dataset.from_pandas(ksof_test),
            })
        loaded_test_wavs = test_datasets.map(prepare_datasets)
        encoded_test_datasets = loaded_test_wavs.map(preprocess_data, batched=True)
        test_metrics = {}
        for m in ['sep', 'fbank', 'ksof']:
            test_metrics[m] = trainer.evaluate(encoded_test_datasets[f'{m}_test'], metric_key_prefix=f'{m}_test')
            trainer.log_metrics(f'{m}_test', test_metrics[m])
        exp_log = {**exp_log, **test_metrics['sep'], **test_metrics['fbank'], **test_metrics['ksof'],
                   'layer': num_hidden_layers}

    else:
        test_metrics = trainer.evaluate(encoded_datasets['test'], metric_key_prefix='test')
        trainer.log_metrics('test', test_metrics)
        trainer.save_metrics(split='all',
                             metrics={**val_metrics, **test_metrics, 'layer': num_hidden_layers})
        exp_log = {**exp_log, **val_metrics, **test_metrics, 'layer': num_hidden_layers}
    logger.info('saving all metrics worked')
    best_model_dir = Path(f'{exp_log_dir}/best_model')
    best_model_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in exp_log.items()])).to_csv(Path(f'{exp_log_dir}/results.csv'),
                                                                               index=False)
    trainer.save_model(str(best_model_dir))
    logger.info(f'{40 * "*"} training {exp_log_dir} done {40 * "*"}')


def norm_wav(wav):
    wav = (wav - np.mean(wav)) / (np.std(wav) + 1e-9)
    return wav


@mlstutterdetection.command()
@click.option('--sep28k-labels', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips.csv')
@click.option('--sep28k-episodes', default='SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv')
@click.option('--fluencybank-labels', default='SEP-28k/ml-stuttering-events-dataset/fluencybank_debug.csv')
@click.option('--audio-dir', default='SEP-28k/ml-stuttering-events-dataset')
@click.option('--ksof-labels', help='ksof label file, need if train data is one of ksof/mix')
@click.option('--ksof-audio-dir',
              help='dir containing the KST dataset with clips, needed when train data is one of ksof/mix')
@click.option('--test-data', help='name of train data to use',
              type=click.Choice(['ksof', 'sep28k', 'fbank', 'sep28fbankfulldev', 'eng', 'smallmixed', 'mixed']),
              default='sep28k')
@click.option('--train-data', help='ksof or sep28k like, 6 or 7 labels',
              type=click.Choice(['ksof', 'sep28k']), default='sep28k')
@click.option("--clf-type", type=click.Choice(['mean', 'deepmean', 'attention', 'sequencetoken']),
              help='which clf version to use')
@click.option('--model', default='facebook/wav2vec2-base-960h', help='path to model/ model name for huggingface')
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--num-hidden-layers', default=12, help='number of transformer layers')
@click.option('--use-weighted-layer-sum', is_flag=True, help='what it says')
@click.option("--stat-pooling", is_flag=True, help='use stats pooling instead of mean pooling')
@click.option('--results-out', default=None, help='path to folder were write predictions and  results')
def make_multilabel_predictions(sep28k_labels, sep28k_episodes, fluencybank_labels, audio_dir,
                                ksof_labels, ksof_audio_dir, test_data, train_data, clf_type, model,
                                stl, num_hidden_layers, use_weighted_layer_sum, stat_pooling, results_out):
    logger = logging.getLogger(f'py.{CLI_NAME}')
    exp_log = {k: v for k, v in locals().items() if not k.startswith('__')}
    transformers.logging.set_verbosity_info()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model, padding=True)
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', padding=True)
    logger.info(feature_extractor)

    if test_data in ['eng', 'smallmixed', 'mixed']:
        _, _, mixed_test = get_combined_train_data(kind=test_data, sep28k_labels=sep28k_labels,
                                                   sep28k_episodes=sep28k_episodes,
                                                   fbank_labels=fluencybank_labels, ksof_labels=ksof_labels,
                                                   sep_audio_dir=audio_dir, ksof_audio_dir=ksof_audio_dir,
                                                   get_test_data=False)
        datasets = DatasetDict(
            {
                f'{test_data}_test': Dataset.from_pandas(mixed_test),
            })
    else:
        sep_test, fb_test, ksof_test = get_combined_train_data(kind=test_data, sep28k_labels=sep28k_labels,
                                                               sep28k_episodes=sep28k_episodes,
                                                               fbank_labels=fluencybank_labels, ksof_labels=ksof_labels,
                                                               sep_audio_dir=audio_dir, ksof_audio_dir=ksof_audio_dir,
                                                               get_test_data=True)
        datasets = DatasetDict(
            {
                'sep_test': Dataset.from_pandas(sep_test),
                'fbank_test': Dataset.from_pandas(fb_test),     # TODO DEBUG
                'ksof_test': Dataset.from_pandas(ksof_test),
            })

    label_cols = [f'is_{col}' for col in LabelLoader().stuttering_cols]
    label_cols = [*label_cols, 'is_Modified'] if train_data == 'ksof' else label_cols

    def prepare_datasets(example):
        _, wav = wavfile.read(example['path'])
        example['audio'] = norm_wav(wav)
        example['label'] = torch.Tensor([example[col] for col in label_cols])
        example['aux_labels'] = 0   # only placeholder
        return example

    def preprocess_data(examples):
        audios = [x for x in examples['audio']]

        inputs = feature_extractor(
            audios,
            sampling_rate=16000,
            max_length=int(3.0 * 16000),
            truncation=True,
            padding=True
        )
        return inputs

    loaded_wavs = datasets.map(prepare_datasets)
    encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)

    config = Wav2Vec2Config.from_pretrained(model)

    if clf_type == 'deepmean':
        seq_clf = Wav2Vec2ForSequenceClassificationAuxDeep.from_pretrained(
            model, ignore_mismatched_sizes=False, config=config, num_aux_labels=2)
    elif clf_type == 'attention':
        seq_clf = Wav2Vec2ForSequenceClassificationCLSAux.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config,
            num_aux_labels=2,
            loss=MultiClassMTLFocalLoss(0.9, 0.7, 3, torch.Tensor([0.5, 0.5]), 2), pooling_mechanism='attention')
    elif clf_type == 'sequencetoken':
        seq_clf = Wav2Vec2ForSequenceClassificationCLSAux.from_pretrained(
            model, ignore_mismatched_sizes=True,
            config=config, num_aux_labels=2)
    else:
        seq_clf = Wav2Vec2ForSequenceClassificationAux.from_pretrained(
            model, ignore_mismatched_sizes=True, num_aux_labels=2)

    predictions, labels, metrics = {}, {}, {}
    with torch.no_grad():
        for m, ds in encoded_datasets.items():
            trainer = Trainer(seq_clf, TrainingArguments(output_dir=results_out, per_device_eval_batch_size=16),
                              tokenizer=feature_extractor,
                              compute_metrics=compute_multilabel_metrics)

            results = trainer.predict(test_dataset=ds, metric_key_prefix=m)
            predictions[m] = results.predictions
            labels[m] = results.label_ids[0]
            metrics = {**metrics, **results.metrics}

    col_names = LabelLoader().stuttering_cols if len(results.predictions[0]) == 6 else LabelLoader().non_fluent_cols

    df_pred = pd.DataFrame()
    for k, v in predictions.items():
        df_pred_tmp = pd.DataFrame(v)
        df_pred_tmp.columns = [f'{c}_pred' for c in col_names]
        labs = labels[k][0] if isinstance(labels[k], tuple) else labels[k]
        df_labels_tmp = pd.DataFrame(labs)
        df_labels_tmp.columns = [f'{c}_label' for c in col_names]
        df_labels_tmp['dataset'] = k
        df_labels_tmp = pd.concat([df_pred_tmp, df_labels_tmp], axis=1)
        df_pred = pd.concat([df_pred, df_labels_tmp])

    results = pd.DataFrame(metrics, index=[0])
    df_pred['model'] = model
    results['model'] = model
    prefix = Path(model).parent.name if Path(model).exists() else model
    df_pred.to_csv(Path(results_out) / f'{prefix}_{test_data}_predictions.csv', index=False)
    results.to_csv(Path(results_out) / f'{prefix}_{test_data}_results.csv', index=False)
    return metrics


@mlstutterdetection.command()
@click.option('--ksof-labels', required=True)
@click.option('--audio-dir', required=True)
@click.option('--stl', is_flag=True, help='ignore auxiliary loss, use stl learning with single CRE loss')
@click.option('--label-col', default='any', help='label col for the run')
@click.option('--aux-col', default='gender', help='auxiliary target for MTL learning')
@click.option('--epochs', default=10, help='number of epochs to train')
@click.option('--early-stopping', default=3, help='number of epochs allowed for eval loss to worsen')
@click.option('--weights', default='superb/hubert-base-superb-ks', help='model training dir, with trainer_state.json, loads best checkpoint')
@click.option('--gradient-cum-steps', default=5, help='Will influence the actual batch size, will be this times 32')
@click.option('--batch-size', default=32, help='number of samples per batch (actual, times gradient-cum-steps)')
@click.option("--log-dir", default='/tmp')
def train_ksof_hubert_classifier(ksof_labels, audio_dir, stl, label_col, aux_col, epochs, early_stopping,
                                 weights, batch_size, gradient_cum_steps, log_dir):
    logger = logging.getLogger(f'py.{CLI_NAME}')

    label_df = load_labels(ksof_labels, 'ksof', audio_dir=audio_dir)

    datasets = DatasetDict(
        {'train': Dataset.from_pandas(label_df[label_df['speaker'].isin(TRAIN_SPK)]),
         'test': Dataset.from_pandas(label_df[label_df['speaker'].isin(TEST_SPK)]),
         'validation': Dataset.from_pandas(label_df[label_df['speaker'].isin(VAL_SPK)])
         })
    # classifier projection size
    # use_weighted_layer_sum
    # consider label smoothing -> not so confident model, helps to tune a better decision boundary
    aux_labels = pd.Series(datasets['train'][aux_col]).value_counts()

    label2id, id2label = {}, {}
    for i, label in enumerate(aux_labels.items()):
        label2id[label[0]] = i
        id2label[i] = label[0]

    def prepare_datasets(example):
        _, wav = wavfile.read(example['path'])
        example['audio'] = norm_wav(wav)
        example['label'] = int(example[label_col])
        if not stl:
            example['aux_labels'] = label2id.get(example[aux_col], 0)
        return example

    loaded_wavs = datasets.map(prepare_datasets)

    now = dt.now()
    time_stamp = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    training_name = f'{time_stamp}_{label_col}_{aux_col}_hubert_ksof_finetuned' if not stl else f'{time_stamp}_{label_col}_hubert_ksof_finetuned_stl'
    exp_log_dir = f'{log_dir}/{label_col}/{training_name}'

    args = TrainingArguments(
        output_dir=exp_log_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,  # 32 is the max the gpus can take
        gradient_accumulation_steps=gradient_cum_steps,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        push_to_hub=False,
        logging_dir=exp_log_dir
    )
    # model_str = get_best_weights_from_transformer_dir(
    #     weights) if weights is not None else 'facebook/wav2vec2-base-960h'

    feature_extractor = AutoFeatureExtractor.from_pretrained(weights)
    def preprocess_data(examples):
        audios = [x for x in examples['audio']]
        inputs = feature_extractor(
            audios,
            sampling_rate=16000,
            max_length=int(3.0 * 16000),
            truncation=True,
            padding=True
        )
        return inputs

    encoded_datasets = loaded_wavs.map(preprocess_data, batched=True)
    class_weights = torch.Tensor(
        1 - pd.Series(encoded_datasets['train'][label_col]).value_counts(normalize=True).sort_index())
    logger.info(f'class weights: {class_weights}')

    seq_clf = HubertForSequenceClassification.from_pretrained(weights)

    logger.info(seq_clf)
    logger.info(f'Model parameters {sum(p.numel() for p in seq_clf.parameters())}')
    logger.info(f'Trainable model parameters {sum(p.numel() for p in seq_clf.parameters() if p.requires_grad)}')

    trainer = Trainer(
        seq_clf,
        args,
        train_dataset=encoded_datasets['train'],
        eval_dataset=encoded_datasets['validation'],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)]
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(encoded_datasets['test'], metric_key_prefix='test')
    trainer.log_metrics('eval', val_metrics)
    trainer.log_metrics('test', test_metrics)
    trainer.save_metrics(split='all', metrics={**val_metrics, **test_metrics})
    trainer.save_model(exp_log_dir)
