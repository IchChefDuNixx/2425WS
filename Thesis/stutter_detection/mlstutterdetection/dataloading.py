import logging
from librosa import yin, pyin
import kaldi_io
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, ComputeDeltas, AmplitudeToDB
import torchaudio
from joblib import Parallel, delayed
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Iterable 
import opensmile


TRAIN_SPK = ('AAV', 'AAW', 'AAZ', 'AIA', 'BAH', 'BSA', 'BTA', 'BUA', 'BWA', 'BBA', 'AAJ')
TEST_SPK = ('AAA', 'AAB', 'AAC', 'AAR', 'AGA', 'AHA', 'ASA', 'AUA', 'BAB', 'BAI', 'ARA', 'AQA', 'BDA', 'AAO')
VAL_SPK = ('AAI', 'AAX', 'AAY', 'ADA', 'AOA', 'ATA', 'BAE', 'BCA', 'BMA', 'AAQ', 'BXA', 'ABA')

DEBUG_SPK = ('AAV', 'AAW', 'AAA', 'AAB' 'AAI', 'AAX')

CV_TRAIN_SPK = ['AAI', 'ABA', 'AAX', 'BBA', 'ADA', 'AAO', 'BAE', 'AAB', 'AAZ', 'BAI', 'ASA', 'AQA', 'AAW', 'BUA', 'AAJ', 'BTA', 'AHA', 'BAH', 'AAC', 'AUA', 'BAB', 'BSA']
CV_VAL_SPK = ['BCA', 'AAA', 'AAQ', 'BXA', 'BDA', 'BMA', 'AIA']
CV_TEST_SPK = ['AAR', 'AAV', 'AAY', 'AGA', 'AOA', 'ATA', 'BWA', 'ARA']

CV_TRAIN_DEV_SPK = CV_TRAIN_SPK + CV_VAL_SPK


class Sep28MelSpectrogramSequenceDataset(Dataset):
    def __init__(self, df_labels, label_col, nj=4, to_db=False, normalize=False):
        mel_spec_config = {'sample_rate': 16000,
                           'n_fft': 400,
                           'win_length': 400,  # 16000 * 0.025
                           'hop_length': 160,  # 16000 * 0.010 -> try different ones 250?
                           'f_min': 0.,
                           'f_max': 8000,
                           'pad': 0,
                           'n_mels': 40,
                           'window_fn': torch.hann_window,
                           'power': 2.,
                           'normalized': normalize,
                           'wkwargs': None,
                           'center': True,
                           'pad_mode': "reflect",
                           'onesided': True,
                           'norm': None,
                           'mel_scale': "htk"}
        self.class_dist = dict(df_labels[label_col].astype(int).value_counts())
        self.labels = list(df_labels[label_col].astype(int))
        self.meta_df = df_labels

        # noinspection PyUnresolvedReferences
        def load_transform(path, a_to_db=False):
            wav, _ = torchaudio.load(path)
            mel_transform = MelSpectrogram(**mel_spec_config)  # returns channel x feature x sequence_length
            spec = mel_transform(wav)
            if a_to_db:
                spec = AmplitudeToDB()(spec)
            spec = spec.transpose(1, 2)
            spec = spec.squeeze(0)
            return spec

        self.data = Parallel(n_jobs=nj)(delayed(load_transform)(p, to_db) for p in list(df_labels['path']))

        self.means = torch.cat(self.data).mean(0)
        self.stds = torch.cat(self.data).std(0)

        # TODO:  gaussian noise, background noise, reverb, speed perturb (0.9, 1.1)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def get_class_dist(self) -> dict:
        return self.class_dist

    def mean_normalize(self, means=None, stds=None):
        if means is None or stds is None:
            self.data = list(map(lambda x: (x - self.means) / self.stds, self.data))  # map object ist not subscriptable
        else:
            self.data = list(map(lambda x: (x - means) / stds, self.data))  # TODO: are mean values correct?

    def get_mean_stats(self):
        return self.means, self.stds

    def get_class_weights(self):
        # TODO: check freaking weights, add external large weights
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_class_dist().items()])
        # {0: 200, 1: 300} -> self._len_ = 500, -> [(1 - (200/500), (1 - (300/500)], [3/5, 2/5]

    def to_pickle(self):
        pass

    @staticmethod
    def from_pickle(pickle_path):
        pass

    def change_label_col(self, label_col):
        self.labels = list(self.meta_df[label_col].astype(int))
        self.class_dist = dict(self.meta_df[label_col].astype(int).value_counts())


class MTLSep28MelSpectrogramSequenceDataset(Sep28MelSpectrogramSequenceDataset):
    def __init__(self, df_labels, label_col, mtl_col='gender', nj=4, to_db=False, normalize=False):
        super(MTLSep28MelSpectrogramSequenceDataset, self).__init__(df_labels, label_col, nj=nj, to_db=to_db,
                                                                    normalize=normalize)
        int_lbls = df_labels[mtl_col].value_counts()
        self.aux_labels = list(df_labels[mtl_col].apply(lambda x: int_lbls.index.get_loc(x)))
        self.aux_class_dist = dict(int_lbls)

    def __getitem__(self, index):
        return self.data[index], (self.labels[index], self.aux_labels[index])

    def get_aux_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_aux_class_dist().items()])
        # {0: 200, 1: 300} -> self._len_ = 500, -> [(1 - (200/500), (1 - (300/500)], [3/5, 2/5]

    def get_aux_class_dist(self) -> dict:
        return self.aux_class_dist


class Sep28MelPitchSequenceDataset(Dataset):
    def __init__(self, df_labels, label_col, nj=4, to_db=True, normalize=False):
        mel_spec_config = {'sample_rate': 16000,
                           'n_fft': 1024,
                           'win_length': 400,  # 16000 * 0.025
                           'hop_length': 160,  # 16000 * 0.010
                           'f_min': 0.,
                           'f_max': 8000,
                           'pad': 0,
                           'n_mels': 40,
                           'window_fn': torch.hann_window,
                           'power': 2.,
                           'normalized': normalize,
                           'wkwargs': None,
                           'center': True,
                           'pad_mode': "reflect",
                           'onesided': True,
                           'norm': None,
                           'mel_scale': "htk"}
        self.labels = list(df_labels[label_col].astype(int))
        mel_spec = MelSpectrogram(**mel_spec_config)  # returns channel x feature x sequence_length
        delta = ComputeDeltas()
        self.class_dist = dict(df_labels[label_col].astype(int).value_counts())

        self.meta_df = df_labels

        # noinspection PyUnresolvedReferences
        def load_transform(path, a_to_db=True):
            """
            format returned resembles image, as we use convolutions as the first nn operations on this (batch first)
            """
            wav, _ = torchaudio.load(path)
            spec = mel_spec(wav)
            if a_to_db:
                spec = AmplitudeToDB()(spec)
            spec = spec.transpose(1, 2).squeeze(0)
            f0, voiced_flag, voiced_prob = pyin(wav.squeeze().numpy(), fmin=70.0, fmax=200, sr=16000, hop_length=160)
            f0_delta = delta(torch.Tensor(f0))  # TODO: is f0 really 0
            f0_feats = torch.stack([torch.Tensor(f0), f0_delta, torch.Tensor(voiced_prob)])
            f0_feats = f0_feats.transpose(0, 1)
            combined = torch.cat([spec, f0_feats], dim=1).unsqueeze(0)

            return combined.nan_to_num()    # TODO figure out what to do with nans

        self.data = Parallel(n_jobs=nj, verbose=1)(delayed(load_transform)(p, to_db) for p in list(df_labels['path']))

        self.means = torch.cat(self.data, dim=1).mean(dim=1).squeeze()
        self.stds = torch.cat(self.data, dim=1).std(dim=1).squeeze()

        # TODO:  gaussian noise, background noise, reverb, speed perturb (0.9, 1.1)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def get_class_dist(self):
        return self.class_dist

    def mean_normalize(self, means=None, stds=None):
        if means is None or stds is None:
            self.data = list(map(lambda x: (x - self.means) / self.stds, self.data))  # map object ist not subscriptable
        else:
            self.data = list(map(lambda x: (x - means) / stds, self.data))  # TODO: are mean values correct?

    def get_mean_stats(self):
        return self.means, self.stds

    def get_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_class_dist().items()])

    def change_label_col(self, label_col):
        self.labels = list(self.meta_df[label_col].astype(int))
        self.class_dist = dict(self.meta_df[label_col].astype(int).value_counts())


class Sep28kWavDataset(Dataset):
    def __init__(self, df_labels, label_col, nj=4):
        self.labels = list(df_labels[label_col].astype(int))
        self.class_dist = dict(df_labels[label_col].astype(int).value_counts())
        self.meta_df = df_labels

        # load wavs to memory
        data = Parallel(n_jobs=nj, verbose=1)(delayed(torchaudio.load)(p) for p in list(df_labels['path']))
        self.data = [w for w, _ in data]
        # TODO:  gaussian noise, background noise, reverb, speed perturb (0.9, 1.1)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def get_class_dist(self):
        return self.class_dist

    def get_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_class_dist().items()])

    def change_label_col(self, label_col):
        self.labels = list(self.meta_df[label_col].astype(int))
        self.class_dist = dict(self.meta_df[label_col].astype(int).value_counts())


class VectorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.from_numpy(d).float() for d in data]
        self.class_dist = dict(pd.Series(labels).astype(int).value_counts())
        self.labels = list(pd.Series(labels).astype(int))
        # print(self.data.size())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def get_class_dist(self) -> dict:
        return self.class_dist

    def get_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_class_dist().items()])


class Wav2VecSequenceDataset(Dataset):
    def __init__(self, df_labels, label_col, extract_layer=1, nj=4):
        self.class_dist = dict(df_labels[label_col].astype(int).value_counts())
        self.labels = list(df_labels[label_col].astype(int))
        self.meta_df = df_labels
        self.extract_layer = extract_layer
        from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
        model = WAV2VEC2_ASR_BASE_960H.get_model()
        # self.model = self.model.eval()

        # noinspection PyUnresolvedReferences
        def load_transform(path, model, device=None, num_layers=12):
            wav, _ = torchaudio.load(path)
            if device is not None:
                wav = wav.to(device)
            features, _ = model.extract_features(wav, num_layers=num_layers)
            return features     # list of tensors, each list item for tensor for one layer
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     self.model = self.model.to(device)
        #     self.features = [load_transform(p, self.model, device) for p in df_labels['path']]
        # else:
        self.features = Parallel(n_jobs=nj, verbose=10)(
                delayed(load_transform)(p, self.model.eval()) for p in list(df_labels['path']))
        self.data = [f[self.extract_layer].squeeze(0) for f in self.features]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def get_class_dist(self) -> dict:
        return self.class_dist

    def change_label_col(self, label_col):
        self.labels = list(self.meta_df[label_col].astype(int))
        self.class_dist = dict(self.meta_df[label_col].astype(int).value_counts())

    def change_extract_layer(self, extract_layer):
        self.extract_layer = extract_layer
        self.data = [f[self.extract_layer].squeeze(0).detach() for f in self.features]

    def get_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_class_dist().items()])


class MTLWav2VecSequenceDataset(Wav2VecSequenceDataset):
    def __init__(self, features, df_labels, label_col, extract_layer, mtl_col='gender'):
        # (df_labels, features) = torch.load(path)

        self.extract_layer = extract_layer
        self.features = features
        self.class_dist = dict(df_labels[label_col].astype(int).value_counts())

        self.labels = list(df_labels[label_col].astype(int))

        int_lbls = df_labels[mtl_col].value_counts()
        self.aux_labels = list(df_labels[mtl_col].apply(lambda x: int_lbls.index.get_loc(x)))
        self.aux_class_dist = dict(int_lbls)

        self.data = [f[self.extract_layer].squeeze(0) for f in self.features]
        self.meta_df = df_labels

    def __getitem__(self, index):
        return self.data[index], (self.labels[index], self.aux_labels[index])

    def get_aux_class_weights(self):
        return torch.Tensor([1 - (v / self.__len__()) for _, v in self.get_aux_class_dist().items()])
        # {0: 200, 1: 300} -> self._len_ = 500, -> [(1 - (200/500), (1 - (300/500)], [3/5, 2/5]

    def get_aux_class_dist(self) -> dict:
        return self.aux_class_dist


class OpenSmileFeatures:
    def __init__(self, label_file, audio_path=None, feature_path=None, normalize=True, nj=1) -> None:
        if audio_path is None and feature_path is None:
            raise AssertionError(f'one of audio_path or feature_path must not be None')
        self.meta = label_file if type(label_file) is pd.DataFrame else pd.read_csv(label_file)
        if feature_path is not None:
            self.features = pd.read_pickle(feature_path)
        if audio_path is not None:
            wavs = [w for w in Path(audio_path).glob('**/*.wav')]
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals)
            features = Parallel(n_jobs=nj, verbose=1)(delayed(get_open_smile_features_file)(w, smile) for w in wavs)
            features = pd.concat(features).reset_index(drop=True)
            features = features.drop(['start', 'end'], axis=1)

            if normalize:
                numeric_cols = features.select_dtypes(include=['float32']).columns
                features[numeric_cols] = features[numeric_cols].apply(
                    lambda x: (x - x.mean()) / x.std(), axis=0)

            self.features = features
            self.features['segment'] = self.features['file'].apply(lambda x: Path(x).stem)
            self.features['speaker'] = self.features['segment'].apply(lambda x: x.split('_')[0])
            # self.combined_df = missing segment_id for
            # join operation -> change all names for challenge, make very simple
            # need segment_nr information

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return f'opensmile data features and metadata for {len(self.meta)} files'

    def persist_features(self, out_path):
        pd.to_pickle(self.features, out_path)

    def decorrelate_feats(self):
        self.features = remove_highly_correlated_feats(self.features, exclude_cols=('file', 'segment'), thresh=0.9)


def get_open_smile_features_file(wav_path: Union[str, Path], smile=None) -> pd.DataFrame:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) if smile is None else smile
    ret = smile.process_file(wav_path)
    return ret.reset_index()


def get_open_smile_features_signal(wav, start, end, smile=None, sr=16000):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) if smile is None else smile
    ret = smile.process_signal(wav, sampling_rate=sr, start=start, end=end)
    return ret.reset_index()


def get_open_smile_features_intervals(a: Union[str, Path], intervals: list, smile=None):
    from scipy.io import wavfile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    ) if smile is None else smile
    sr, wav = wavfile.read(a)
    df = pd.DataFrame()
    for inter in intervals:
        start, end = inter
        os = get_open_smile_features_signal(wav, start, end, smile=smile, sr=sr)
        os['name'] = f'{start}_{end}_{str(a)}'
        df = df.append(os)
    return df


def load_open_smile_data(audios: Iterable, nj=1):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    res = Parallel(n_jobs=nj)(delayed(get_open_smile_features_file)(a, smile) for a in tqdm(audios))
    df = pd.DataFrame()
    for data in res:
        df = df.append(data, ignore_index=True)
    return df


def remove_highly_correlated_feats(df, exclude_cols=('label', 'file'), thresh=0.9, return_cols=False):
    numeric_cols = df.columns.difference(exclude_cols)
    # corr_mat = df[numeric_cols].corr() # this is bullshit, don't do this
    corr_mat = np.corrcoef(df[numeric_cols].to_numpy(), rowvar=False)
    corr_feats = set()
    for i in range(1, len(numeric_cols)):
        for j in range(1, i):
            if abs(corr_mat[i, j]) > thresh:
                corr_feats.add(numeric_cols[i])
    numeric_cols = numeric_cols.drop(corr_feats)

    return [*exclude_cols, *numeric_cols] if return_cols else df[[*exclude_cols, *numeric_cols]]


def remove_highly_correlated_parr(df, exclude_cols=('label', 'file'), thresh=0.9, nj=4) -> pd.DataFrame:
    numeric_cols = df.columns.difference(exclude_cols)
    corr_mat = df[numeric_cols].corr()  # vermutlich bottleneck
    corr_feats = set()

    def get_corrs(i):
        tmp_feats = set()
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > thresh:
                tmp_feats.add(corr_mat.columns[i])
        return tmp_feats

    res = Parallel(n_jobs=nj)(delayed(get_corrs)(x) for x in tqdm(range(len(corr_mat.columns))))
    for s in res:
        corr_feats.update(s)
    numeric_cols = numeric_cols.drop(corr_feats)
    return df[[*exclude_cols, *numeric_cols]]


def load_labels(label_file, kind='sep28k', audio_dir=None, **kwargs):
    datasets = {
        'sep28k': Sep28kLabelLoader(),
        'fluencybank': FluencyBankLabelLoader(),
        'ksof': KSOFLabelLoader(),
        }

    assert(kind in datasets.keys())

    return datasets[kind].load_labels(label_file=label_file, audio_dir=audio_dir, **kwargs)


def load_splits(label_file, kind='sep28k', audio_dir=None, **kwargs):
    datasets = {
        'sep28k': Sep28kLabelLoader(),
        'fluencybank': FluencyBankLabelLoader(),
        'ksof': KSOFLabelLoader(),
        }

    assert(kind in datasets.keys())

    return datasets[kind].load_splits(label_file=label_file, audio_dir=audio_dir, **kwargs)


class LabelLoader:
    def __init__(self):
        self.non_fluent_cols = ['SoundRep', 'Prolongation', 'Block',
                                'WordRep', 'Interjection', 'NoStutteredWords', 'Modified']
        self.ksof_cols = ['Sound Repetition', 'Prolongation', 'Block', 'Word / Phrase Repetition', 'Interjection',
                          'No dysfluencies', 'Modified/ Speech technique']
        self.col_mapping = {k: v for k, v in zip(self.ksof_cols, self.non_fluent_cols)}
        self.stuttering_cols = self.non_fluent_cols[0:-1]   # modified is not stuttering and not included in sep28k/fb

    def load_splits(self, label_file, audio_dir=None, **kwargs):
        raise NotImplementedError

    def load_labels(self, label_file, audio_dir=None):
        raise NotImplementedError

    def make_int_label(self, labels, lbl_cols):
        labels['clean_label'] = labels.apply(lambda x: x[lbl_cols].sum() == 1, axis=1)
        labels['int_label'] = np.nan
        for col in lbl_cols:
            labels.loc[labels['clean_label'], 'int_label'] = labels.loc[labels['clean_label']].apply(
                lambda x: lbl_cols.index(col) if x[col] else x['int_label'], axis=1)
        return labels


class CompareLabelLoader:
    def __init__(self, challenge_dir):
        """
        folder structure of challenge package
        .
        ├── features
            ├── audeep
            ├── deepspectrum
            ├── opensmile
            └── xbow
        ├── lab
        └── wav
        """
        super().__init__()
        self.challenge_dir = Path(challenge_dir)
        label_dir = self.challenge_dir / 'lab'
        labels = pd.DataFrame()
        for csv in Path(f'{label_dir}').glob('**/*.csv'):
            lab = pd.read_csv(csv)
            lab['partition'] = csv.stem
            labels = pd.concat([labels, lab])
        labels.loc[labels['partition'] == 'devel', 'partition'] = 'dev'
        labels.reset_index(drop=True, inplace=True)
        labels['path'] = labels['filename'].apply(lambda x: f'{self.challenge_dir}/wav/{x}')
        labels['any'] = labels['label'].apply(lambda x: x not in ['Garbage', 'no_disfluencies'])
        labels['dataset'] = 'ksfc'
        labels['language'] = 'de'

        self.fluencybank_labels = None
        self.sep_labels = None
        self.labels = labels
        self.label_mapping = self.labels['label'].unique().tolist()
        self.label_mapping.sort()
        self.labels['label_name'] = self.labels['label']
        self.labels['label'] = self.labels['label'].apply(lambda x: self.label_mapping.index(x))

    def load_splits(self, split='ksof_c', oversample_ksf_c_wd=None):
        train = self.labels.loc[self.labels['partition'] == 'train'].reset_index(drop=True)
        dev = self.labels.loc[self.labels['partition'] == 'dev'].reset_index(drop=True)
        test = self.labels.loc[self.labels['partition'] == 'test'].reset_index(drop=True)

        if oversample_ksf_c_wd is not None:
            from math import floor
            oversample_ksf_c_wd = float(oversample_ksf_c_wd)
            wds = train.loc[train['label_name'] == 'WordRepetition']
            vc = self.labels['label_name'].value_counts(normalize=True)
            number_of_reps = floor(vc[0] * oversample_ksf_c_wd / vc['WordRepetition'])
            wds = pd.concat([wds for _ in range(number_of_reps)])
            train = pd.concat([train, wds])
            train = train.sample(frac=1).reset_index(drop=True)
        if split == 'ksof_c':
            return train, dev, test
        else:
            if (self.fluencybank_labels is None) | (self.sep_labels is None):
                raise NotImplementedError('please load fluencybank and the sep labels first')
        if split in ['sep28k_c', 'eng_c']:
            # potentially bad, if reuse of object is intended, but quick & dirty fix for pre-training
            # logging output will be wrong, but is only for pre-training, so ignore
            self.fluencybank_labels['label'] = self.fluencybank_labels['label'].apply(lambda x: x - 1 if x > 3 else x)
            self.sep_labels['label'] = self.sep_labels['label'].apply(lambda x: x - 1 if x > 3 else x)
        # ['Block', 'Fillers', 'Garbage', 'Modified', 'Prolongation', 'SoundRepetition', 'WordRepetition',
        #         'no_disfluencies']

        fb_train = self.fluencybank_labels[self.fluencybank_labels['partition'] == 'train'].reset_index(drop=True)
        fb_dev = self.fluencybank_labels[self.fluencybank_labels['partition'] == 'dev'].reset_index(drop=True)
        fb_test = self.fluencybank_labels[self.fluencybank_labels['partition'] == 'test'].reset_index(drop=True)
        sep_train = self.sep_labels[self.sep_labels['partition'] == 'train'].reset_index(drop=True)
        sep_dev = self.sep_labels[self.sep_labels['partition'] == 'dev'].reset_index(drop=True)
        sep_test = self.sep_labels[self.sep_labels['partition'] == 'test'].reset_index(drop=True)

        train_garbage = pd.concat(
            [fb_train[fb_train['label_name'] == 'Garbage'], sep_train[sep_train['label_name'] == 'Garbage'],
             fb_dev[fb_dev['label_name'] == 'Garbage'], sep_dev[sep_dev['label_name'] == 'Garbage']])

        dev_garbage = pd.concat(
            [sep_test[sep_test['label_name'] == 'Garbage'], fb_test[fb_test['label_name'] == 'Garbage']])

        mod_train = train.loc[train['label_name'] == 'Modified']
        mod_dev = train.loc[train['label_name'] == 'Modified']

        splits = {
            'ksof_c_garbage': (pd.concat([train, train_garbage]).sample(frac=1).reset_index(drop=True),
                               pd.concat([dev, dev_garbage]).sample(frac=1).reset_index(drop=True), test),
            'sep28k_c': (sep_train, sep_dev, sep_test),
            'fb_c': (fb_train, fb_dev, fb_test),
            'sep28k_c_m': (pd.concat([sep_train, mod_train, sep_dev]).sample(frac=1).reset_index(drop=True),
                           pd.concat([sep_test, fb_test, mod_dev]).sample(frac=1).reset_index(drop=True), test),
            'eng_c': (pd.concat([sep_train, fb_train]).sample(frac=1).reset_index(drop=True),
                      pd.concat([sep_dev, fb_dev]).sample(frac=1).reset_index(drop=True),
                      pd.concat([sep_test, fb_test]).sample(frac=1).reset_index(drop=True)),
            'eng_c_m': (
                pd.concat([sep_train, mod_train, sep_dev, fb_train, fb_dev]).sample(frac=1).reset_index(drop=True),
                pd.concat([sep_test, mod_dev, fb_test]).sample(frac=1).reset_index(drop=True), test),
            'full': (
                pd.concat([train, sep_train, sep_dev, fb_train, fb_dev]).sample(frac=1).reset_index(drop=True),
                pd.concat([sep_test, fb_test, dev]).sample(frac=1).reset_index(drop=True), test),
            'full_dev_gerfb': (
                pd.concat([train, sep_train, sep_dev, fb_train, fb_dev, sep_test]).sample(frac=1).reset_index(drop=True),
                pd.concat([fb_test, dev]).sample(frac=1).reset_index(drop=True), test),
            'full_dev_ger': (
                pd.concat([train, sep_train, sep_dev, fb_train, fb_dev, fb_test, sep_test]).sample(frac=1).reset_index(
                    drop=True), dev, test)
        }
        return splits[split]

    def load_labels(self):
        return self.labels

    def idx_to_label(self, idx):
        if idx not in range(0, len(self.label_mapping)):
            raise ValueError(f'Idx {idx} not in labels')
        return self.label_mapping[idx]

    def label_to_idx(self, label):
        if label not in self.label_mapping:
            raise ValueError(f'Label {label} not in labels')
        return self.label_mapping.index(label)

    def __transform_to_compare(self, labels):
        labels.drop(columns=['Block', 'Prolongation'], inplace=True)    # otherwise duplicate columns later
        label_cols = ['is_Block', 'is_Interjection', 'is_Garbage', 'is_Modified', 'is_Prolongation', 'is_SoundRep',
                      'is_WordRep', 'is_NoStutteredWords']
        labels = labels.loc[labels[label_cols].sum(axis=1) < 2].reset_index(drop=True)
        translate_cols = {k: v for k, v in zip(label_cols, self.label_mapping)}
        translate_cols['SEP28k-E'] = 'partition'
        labels.rename(translate_cols, inplace=True, axis=1)
        labels['label_name'] = labels[self.label_mapping].idxmax(axis=1)       # weird pandas shit, idx is a string
        labels['label'] = labels['label_name'].apply(lambda x: self.label_to_idx(x))
        labels['filename'] = labels['path'].apply(lambda x: Path(x).name)

        return labels[self.labels.columns]

    def load_sep28k(self, label_file, audio_dir, extended_episodes_file, no_wd=False):
        labels = load_labels(label_file=label_file, kind='sep28k', extended_episodes_file=extended_episodes_file,
                             audio_dir=audio_dir)
        labels['dataset'] = 'sep28k'
        labels = self.__transform_to_compare(labels)
        if no_wd:
            labels = labels.loc[labels['label_name'] != 'WordRepetition'].reset_index(drop=True)
        labels['language'] = 'eng'
        self.sep_labels = labels
        return labels

    def load_fluencybank(self, label_file, audio_dir, no_wd=False):
        labels = load_labels(label_file=label_file, kind='fluencybank', audio_dir=audio_dir)
        labels['dataset'] = 'fluencybank'
        labels = self.__transform_to_compare(labels)
        if no_wd:
            labels = labels.loc[labels['label_name'] != 'WordRepetition'].reset_index(drop=True)
        labels['language'] = 'eng'
        self.fluencybank_labels = labels
        return labels

    @staticmethod
    def static_cols():
        return ['Block', 'Fillers', 'Garbage', 'Modified', 'Prolongation', 'SoundRepetition', 'WordRepetition',
                'no_disfluencies']


class Sep28kLabelLoader(LabelLoader):
    def __init__(self):
        super().__init__()

    def load_labels(self, label_file, audio_dir=None, extended_episodes_file=None):
        labels = pd.read_csv(label_file)
        labels['utt'] = labels.apply(lambda x: f'{x.Show}_{x.EpId}_{x.ClipId}', axis=1)
        if audio_dir is not None:
            labels['path'] = labels.apply(lambda x: f'{audio_dir}/clips/{x.Show}/{x.EpId}/{x.utt}.wav', axis=1)
        for col in self.stuttering_cols:
            labels[f'is_{col}'] = labels[col].apply(lambda x: x >= 2)

        if extended_episodes_file is not None:
            cols = [*labels.columns, 'gender', 'host_gender', 'guest_gender']
            episodes = pd.read_csv(extended_episodes_file, header=0)
            episodes['real_ep_id'] = (episodes['Show'] + '_' + episodes['EpId'].astype(str)).str.strip()
            labels['real_ep_id'] = (labels['Show'] + '_' + labels['EpId'].astype(str)).str.strip()

            merged = pd.merge(left=labels, right=episodes, left_on='real_ep_id', right_on='real_ep_id', how='inner',
                              suffixes=('', '_drop'))
            merged['gender'] = merged.apply(lambda x: x['host_gender'] if x.is_probably_host else x['guest_gender'],
                                            axis=1)
            labels = merged[cols].copy()
        else:
            # assume gender based on show, clips actually only from 8 episodes?
            lookup = {
                'WomenWhoStutter': 'f',  # host is female, guests are female, same host as HeStutters
                'StutteringIsCool': 'm',  # host is male
                'StutterTalk': 'm',  # has also a female host, but is a minority
                'MyStutteringLife': 'm',  # host is male
                'StrongVoices': 'm',  # hosts are male
                'HeStutters': 'f',  # host is female, interviewees are male
                'IStutterSoWhat': 'm',  # host is male
                'HVSA': 'm'  # host is male
            }
            # Episodes from which the clips are actually taken -> only 8 episodes
            # cool148                                     110   # all male one dude StutteringIsCool
            # 583StutterTalk                               85   # male host, male guest StutterTalk
            # 682StutterTalk                                4   # male host, female guest StutterTalk
            # episode-180-with-petra-a                     82   # all female WomenWhoStutter
            # episode-175-with-rachel-hoge-second-time     40   # all female  WomenWhoStutter
            # male-episode-11-with-frank                   35   # female host, male guest HeStutters
            # episode-208-with-kelsey-h                    24   # all female voices, has echo HeStutters
            # episode-137-with-autumn                       5   # all female voices SheStutters
            # => max 15 speakers
            labels['gender'] = labels['Show'].apply(lambda x: lookup.get(x, 'm'))  # statistically speaking, more males

        lbl_cols = [f'is_{col}' for col in self.stuttering_cols]

        labels['any'] = labels[lbl_cols].apply(lambda x: any(x), axis=1)
        labels['is_Garbage'] = labels[['Unsure', 'PoorAudioQuality', 'DifficultToUnderstand', 'NoSpeech', 'Music']].sum(
            axis=1) >= 2
        labels['is_Modified'] = False
        labels['language'] = 'eng'
        labels = self.make_int_label(labels, lbl_cols)

        return labels

    def load_splits(self, label_file, audio_dir=None, extended_episodes_file=None):
        labels = self.load_labels(label_file, extended_episodes_file=extended_episodes_file, audio_dir=audio_dir)
        return labels.loc[labels['SEP28k-E'] == 'train'].reset_index(drop=True), labels.loc[
            labels['SEP28k-E'] == 'dev'].reset_index(drop=True), labels.loc[labels['SEP28k-E'] == 'test'].reset_index(
            drop=True)


class FluencyBankLabelLoader(LabelLoader):
    def __init__(self):
        super().__init__()
        self.TRAIN_SPK = ['24fa', '62m', '50fa', '29ma', '64m', '35ma', '27ma', '29mb', '28m', 
                          '46mb', '43m', '26f', '60m', '35mb', '62f', '24mb',
                          '25m', '39f', '54f', '50fb']
        self.TEST_SPK = ['68m', '26m', '54m', '24fc', '34m', '24ma']
        self.DEV_SPK = ['46ma', '27f', '24fb', '37m', '29mc', '32m', '57m']
        self.TRAIN_DEV_SPK = self.TRAIN_SPK + self.DEV_SPK

    def load_labels(self, label_file, audio_dir=None):
        labels = pd.read_csv(label_file)
        episodes = Path(label_file).parent / 'fluencybank_episodes.csv'
        lbl_cols = [f'is_{col}' for col in self.stuttering_cols]
        labels['utt'] = labels.apply(lambda x: f'{x.Show}_{x.EpId:03d}_{x.ClipId}', axis=1)
        if audio_dir is not None:
            labels['path'] = labels.apply(lambda x: f'{audio_dir}/clips/{x.Show}/{x.EpId:03d}/{x.utt}.wav', axis=1)

        if episodes.exists():   # this adds gender col to fluencybank
            eps = pd.read_csv(episodes, names=['Show', 'EpId', "url", 'trash1', 'trash2'])
            eps['gender'] = eps['url'].apply(lambda x: 'f' if 'f' in x.split('/')[-1] else 'm')
            eps['speaker'] = eps['url'].apply(lambda x: x.split('/')[-1].split('.')[0])
            eps.drop(['Show', 'url', 'trash1', 'trash2'], axis=1, inplace=True)
            labels = labels.merge(eps, left_on='EpId', right_on='EpId')
        labels['partition'] = labels['speaker'].map(
            {
                **{spk: 'train' for spk in self.TRAIN_SPK},
                **{spk: 'dev' for spk in self.DEV_SPK},
                **{spk: 'test' for spk in self.TEST_SPK},
            })
        for col in self.stuttering_cols:
            labels[f'is_{col}'] = labels[col].apply(lambda x: x >= 2)

        labels['any'] = labels[lbl_cols].apply(lambda x: any(x), axis=1)
        labels['is_Garbage'] = labels[['Unsure', 'PoorAudioQuality', 'DifficultToUnderstand', 'NoSpeech', 'Music']].sum(
            axis=1) >= 2
        labels['is_Modified'] = False
        labels['language'] = 'eng'

        labels = self.make_int_label(labels, lbl_cols)

        return labels
        labels = self.load_labels(label_file, audio_dir=audio_dir)
        return labels.loc[labels.partition == 'train'].reset_index(drop=True), labels.loc[
            labels.partition == 'dev'].reset_index(drop=True), labels.loc[labels.partition == 'test'].reset_index(
            drop=True)


class KSOFLabelLoader(LabelLoader):
    def __init__(self):
        super().__init__()

    def load_labels(self, label_file, audio_dir=None):
        labels = pd.read_csv(label_file)
        name_col = 'segment_name' if 'segment_name' in labels.columns else 'segment_id'

        labels.rename(columns=self.col_mapping, inplace=True)
        non_fluent_cols = [*self.stuttering_cols, 'Modified']
        lbl_cols = [f'is_{col}' for col in non_fluent_cols]

        for col in non_fluent_cols:
            labels[f'is_{col}'] = labels[col].apply(lambda x: x >= 2)

        if audio_dir is not None:
            labels['path'] = labels.apply(lambda x: f'{audio_dir}/segments/{x[name_col]}.wav', axis=1)
            # filter out non existing audio files:
            labels['exists'] = labels['path'].apply(lambda x: Path(x).exists())
            filterd_out = len(labels)
            labels = labels.loc[labels.exists].reset_index(drop=True)
            if filterd_out != len(labels):
                logging.warning(f'Filtered out {filterd_out - len(labels)} non existing audio files.')
        labels['non_fluent'] = labels[lbl_cols].apply(lambda x: any(x), axis=1)
        labels['utt'] = labels[name_col]
        # any only w.r.t to stuttering, ksof has modified extra, so this column is non_fluent
        labels['any'] = labels[[f'is_{col}' for col in self.stuttering_cols]].apply(lambda x: any(x), axis=1)

        labels = self.make_int_label(labels, lbl_cols)
        labels['language'] = 'de'

        return labels

    def load_splits(self, label_file, audio_dir=None, **kwargs):
        labels = self.load_labels(label_file, audio_dir=audio_dir)
        return labels.loc[labels['partition'] == 'train'].reset_index(drop=True), labels.loc[
            labels['partition'] == 'dvel'].reset_index(drop=True), labels.loc[
                   labels['partition'] == 'test'].reset_index(drop=True)


def get_combined_train_data(kind='ksof', sep28k_labels=None, sep28k_episodes=None, fbank_labels=None, ksof_labels=None,
                            sep_audio_dir=None, ksof_audio_dir=None, get_test_data=False):
    """
    :param get_test_data:
    :param kind:
    :param sep28k_labels:
    :param sep28k_episodes:
    :param fbank_labels:
    :param ksof_labels:
    :param sep_audio_dir:
    :param ksof_audio_dir:
    :return: tuple of train, dev, test dataframes, depending on the kind parameter
    """
    sep_train, sep_dev, sep_test, ksof_train, ksof_dev, ksof_test, fb_train, fb_dev, fb_test = None, None, None, None, None, None, None, None, None
    if sep28k_labels is not None:
        sep_train, sep_dev, sep_test = load_splits(kind='sep28k', label_file=sep28k_labels, audio_dir=sep_audio_dir,
                                                   extended_episodes_file=sep28k_episodes)
    if fbank_labels is not None:
        fb_train, fb_dev, fb_test = load_splits(kind='fluencybank', label_file=fbank_labels, audio_dir=sep_audio_dir)

    if ksof_labels is not None:
        ksof_train, ksof_dev, ksof_test = load_splits(kind='ksof', label_file=ksof_labels, audio_dir=ksof_audio_dir)

    if get_test_data:
        return sep_test, fb_test, ksof_test
    else:
        splits = {
            'ksof': (ksof_train, ksof_dev, ksof_test),
            'sep28k': (sep_train, sep_dev, sep_test),
            'fbank': (fb_train, fb_dev, fb_test),
            'sep28fbankfulldev': (pd.concat([sep_train, sep_dev]), pd.concat([fb_train, fb_dev, fb_test]), sep_test),
            'eng': (pd.concat([sep_train, fb_train]), pd.concat([sep_dev, fb_dev]), pd.concat([sep_test, fb_test])),
            'smallmixed': (pd.concat([fb_train, ksof_train]), pd.concat([fb_dev, ksof_dev]),
                           pd.concat([fb_test, ksof_test])),
            'mixed': (pd.concat([sep_train, fb_train, ksof_train]), pd.concat([sep_dev, fb_dev, ksof_dev]),
                      pd.concat([sep_test, fb_test, ksof_test])),
        }

    return splits[kind]


def load_xvectors(scp):
    x_vecs = {key: vec for key, vec in kaldi_io.read_vec_flt_scp(scp)}
    df = pd.DataFrame(x_vecs).T
    df = df.reset_index()
    df['segment_name'] = df['index'].apply(lambda x: x.split('-')[0])
    df = df.drop('index', axis=1)
    df = df.groupby('segment_name').mean()
    df = df.reset_index()
    return df
