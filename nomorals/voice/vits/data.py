"""Dataset pipeline: filelists, loaders, collation, bucket sampling.

Numpy port of upstream ``data_utils.py`` (the torch ``Dataset``/``DataLoader``
wrapping lives in the trainer, which converts these numpy batches to
tensors). Fixes and changes, all recorded in the survey:

* ``TextAudioSpeakerLoader._filter`` upstream never initialises ``n_fau``,
  so it raises ``UnboundLocalError`` on every filelist. Here it is
  initialised to 0. The speaker loader also uses a hardcoded 7.0 s
  duration cap (upstream); the single-speaker loader uses
  ``max_wav_len`` from the config (default 10.0 s) — both preserved.
* Durations come from the WAV header (``audio.wav_duration``) instead of
  ``librosa.get_duration``.
* Spectrogram caches are ``.spec.npz`` (see ``audio.load_or_compute_spec``)
  instead of upstream's ``.spec.pt``.
* Sidecars (``.torchmoji`` / ``.bert``) are loaded with numpy when they
  are ``.npz`` and with torch when they are upstream ``.pt`` files; a
  missing sidecar is the same ``ValueError`` upstream raises.
* ``DistributedBucketSampler`` is the same algorithm, with
  ``numpy.random.Generator`` in place of ``torch.Generator`` for the
  shuffled case; the unshuffled case (pinned in parity) is identical.
"""

import os
import random

import numpy as np

from . import audio as _audio
from . import mathutils
from .text import cleaned_text_to_sequence, text_to_sequence

try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch is optional here
    _torch = None


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = []
        for line in f:
            line_splits = line.strip().split(split)
            if len(line_splits) < 2:
                print(f"Warning: line {line.strip()} invalid split")
                continue
            filepaths_and_text.append(line_splits)
    return filepaths_and_text


def _load_sidecar(path, name):
    """Load a ``.torchmoji`` / ``.bert`` sidecar.

    Numpy sidecars are stored as ``<path>.npz``; upstream torch sidecars
    are ``<path>`` (a ``.pt`` file). Both spellings are accepted.
    """
    if os.path.exists(path + ".npz"):
        path = path + ".npz"
    if not os.path.exists(path):
        raise ValueError(f"{path} {name} doesn't exist!")
    if path.endswith(".npz"):
        with np.load(path) as data:
            keys = list(data.keys())
            return data[keys[0]]
    if _torch is None:
        raise ImportError(
            f"sidecar {path} is a torch .pt file but torch is not installed"
        )
    return _torch.load(path, map_location="cpu", weights_only=False).numpy()


class TextAudioLoader:
    """Loads audio/text pairs, normalises text, computes spectrograms."""

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 250)
        self.max_wav_len = getattr(hparams, "max_wav_len", 10.0)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []
        n_fau = 0
        for current in self.audiopaths_and_text:
            try:
                audiopath, text = current
            except ValueError:
                print(f"{current} failed")
                continue
            au_len = _audio.wav_duration(audiopath)
            if au_len > self.max_wav_len:
                n_fau += 1
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        print(f"{n_fau} removed due to too long")

    def get_audio(self, filename):
        audio = _audio.load_wav_normalized(
            filename, self.sampling_rate, self.max_wav_value)
        spec = _audio.load_or_compute_spec(filename, self._hparams_data())
        return spec, audio

    def _hparams_data(self):
        class _D:
            pass
        d = _D()
        d.sampling_rate = self.sampling_rate
        d.max_wav_value = self.max_wav_value
        d.filter_length = self.filter_length
        d.hop_length = self.hop_length
        d.win_length = self.win_length
        return d

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = mathutils.intersperse(text_norm, 0)
        return np.array(text_norm, dtype=np.int64)

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        torchmoji = self.get_torchmoji(audiopath)
        bert = self.get_bert(audiopath)
        return (text, spec, wav, torchmoji, bert)

    def get_torchmoji(self, audiopath):
        moji_filename = audiopath.replace(".wav", ".torchmoji")
        arr = _load_sidecar(moji_filename, "torchmoji embedding")
        # Upstream squeezes twice: [1, 1, 2304] -> [2304].
        return np.squeeze(arr)

    def get_bert(self, audiopath):
        bert_filename = audiopath.replace(".wav", ".bert")
        arr = _load_sidecar(bert_filename, "bert embedding")
        return np.squeeze(arr, 0)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets (numpy version of upstream)."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        ids_sorted_decreasing = _sort_desc([x[1].shape[1] for x in batch])
        max_text_len = max(len(x[0]) for x in batch)
        max_spec_len = max(x[1].shape[1] for x in batch)
        max_wav_len = max(x[2].shape[1] for x in batch)

        text_lengths = np.zeros(len(batch), dtype=np.int64)
        spec_lengths = np.zeros(len(batch), dtype=np.int64)
        wav_lengths = np.zeros(len(batch), dtype=np.int64)
        moji_dim = batch[0][3].shape[0]
        bert_shape = batch[0][4].shape

        text_padded = np.zeros((len(batch), max_text_len), dtype=np.int64)
        spec_padded = np.zeros((len(batch), batch[0][1].shape[0], max_spec_len), dtype=np.float32)
        wav_padded = np.zeros((len(batch), 1, max_wav_len), dtype=np.float32)
        torchmoji_hidden = np.zeros((len(batch), moji_dim), dtype=np.float32)
        bert_padded = np.zeros((len(batch), bert_shape[0], bert_shape[1]), dtype=np.float32)
        bert_lengths = np.zeros(len(batch), dtype=np.int64)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            spec = row[1]
            spec_padded[i, :, :spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav = row[2]
            wav_padded[i, :, :wav.shape[1]] = wav
            wav_lengths[i] = wav.shape[1]
            torchmoji_hidden[i, :] = row[3]
            bert = row[4]
            bert_padded[i, :bert.shape[0], :] = bert
            bert_lengths[i] = bert.shape[0]

        if self.return_ids:
            return (text_padded, text_lengths, spec_padded, spec_lengths,
                    wav_padded, wav_lengths, torchmoji_hidden, bert_padded,
                    bert_lengths, ids_sorted_decreasing)
        return (text_padded, text_lengths, spec_padded, spec_lengths,
                wav_padded, wav_lengths, torchmoji_hidden, bert_padded,
                bert_lengths)


def _sort_desc(values):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return np.array(order, dtype=np.int64)


class TextAudioSpeakerLoader(TextAudioLoader):
    """Multi-speaker version: filelist rows are ``wav|speaker_id|text``."""

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        audiopaths_sid_text_new = []
        lengths = []
        n_fau = 0
        for audiopath, sid, text in self.audiopaths_sid_text:
            au_len = _audio.wav_duration(audiopath)
            if au_len > 7.0:
                n_fau += 1
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(f"{n_fau} removed for too long.")

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, sid, text = (audiopath_sid_text[0], audiopath_sid_text[1],
                                audiopath_sid_text[2])
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        torchmoji = self.get_torchmoji(audiopath)
        bert = self.get_bert(audiopath)
        sid = np.array([int(sid)], dtype=np.int64)
        return (text, spec, wav, torchmoji, bert, sid)

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets for the multi-speaker loader."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        ids_sorted_decreasing = _sort_desc([x[1].shape[1] for x in batch])
        max_text_len = max(len(x[0]) for x in batch)
        max_spec_len = max(x[1].shape[1] for x in batch)
        max_wav_len = max(x[2].shape[1] for x in batch)

        text_lengths = np.zeros(len(batch), dtype=np.int64)
        spec_lengths = np.zeros(len(batch), dtype=np.int64)
        wav_lengths = np.zeros(len(batch), dtype=np.int64)
        sid = np.zeros(len(batch), dtype=np.int64)
        moji_dim = batch[0][3].shape[0]
        bert_shape = batch[0][4].shape

        text_padded = np.zeros((len(batch), max_text_len), dtype=np.int64)
        spec_padded = np.zeros((len(batch), batch[0][1].shape[0], max_spec_len), dtype=np.float32)
        wav_padded = np.zeros((len(batch), 1, max_wav_len), dtype=np.float32)
        torchmoji_hidden = np.zeros((len(batch), moji_dim), dtype=np.float32)
        bert_padded = np.zeros((len(batch), bert_shape[0], bert_shape[1]), dtype=np.float32)
        bert_lengths = np.zeros(len(batch), dtype=np.int64)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            spec = row[1]
            spec_padded[i, :, :spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav = row[2]
            wav_padded[i, :, :wav.shape[1]] = wav
            wav_lengths[i] = wav.shape[1]
            torchmoji_hidden[i, :] = row[3]
            bert = row[4]
            bert_padded[i, :bert.shape[0], :] = bert
            bert_lengths[i] = bert.shape[0]
            sid[i] = row[5]

        if self.return_ids:
            return (text_padded, text_lengths, spec_padded, spec_lengths,
                    wav_padded, wav_lengths, torchmoji_hidden, bert_padded,
                    bert_lengths, sid, ids_sorted_decreasing)
        return (text_padded, text_lengths, spec_padded, spec_lengths,
                wav_padded, wav_lengths, torchmoji_hidden, bert_padded,
                bert_lengths, sid)


class DistributedBucketSampler:
    """Maintain similar input lengths in a batch (upstream algorithm).

    Boundaries ``[b1, b2, ...]`` group samples by length; samples outside
    the range are discarded. ``shuffle=False`` is deterministic and pinned
    in parity; ``shuffle=True`` uses a numpy Generator seeded by epoch
    (upstream uses a torch Generator, so shuffled permutations differ
    numerically but have the same bucket structure).
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None,
                 rank=None, shuffle=True):
        self.lengths = list(dataset.lengths)
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.shuffle = shuffle
        self.epoch = 0
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(rng.permutation(len(bucket)).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
            rem = num_samples_bucket - len_bucket
            ids_bucket = (ids_bucket + ids_bucket * (rem // len_bucket)
                          + ids_bucket[:(rem % len_bucket)])
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)
        if self.shuffle:
            batch_ids = rng.permutation(len(batches)).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples // self.batch_size
