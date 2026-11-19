"""VITS inference frontend (``VITSFrontend``).

Port of upstream ``tsvitsfe.py`` + the ``infer_ts.py`` CLI, with the
TorchScript model loaded through the same path upstream's
``export_ts.py`` writes. ``infer`` takes raw text and returns a float32
numpy waveform at the config sample rate; the emotion text defaults to
the synthesis text, exactly as upstream.

Requires: torch, a TorchScript-exported model (see ``train.py`` /
``export``), TorchMoji weights, TinyBERT weights. Every missing piece
raises a named error.
"""

import os

import numpy as np

from . import mathutils, train_utils
from ._torch import require_torch
from .configs import load_config
from .text import cleaned_text_to_sequence, text_to_sequence
from .text.cleaners import arpa_cleaners


class VITSFrontend:
    """Load a TorchScript VITS model and synthesize speech from text."""

    def load(self, model_path, config_path, device="cpu",
             bert_model_name=None, bert_cache_dir=None):
        torch = require_torch("VITS inference")
        from .bertfe import DEFAULT_MODEL, BERTFrontEnd
        from .moji import TorchMoji
        from .stft import TorchSTFT

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"TorchScript VITS model not found at {model_path}; export one first "
                "(see nomorals.voice.vits.train export)"
            )
        self.model = torch.jit.load(model_path).to(device)
        self.moji = TorchMoji()
        self.bert_f = BERTFrontEnd("cuda" in device,
                                   model_name=bert_model_name or DEFAULT_MODEL,
                                   cache_dir=bert_cache_dir)
        self.hps = load_config(config_path)
        self.stft = TorchSTFT(self.hps.model.gen_istft_n_fft,
                              hop_length=self.hps.model.gen_istft_hop_size,
                              win_length=self.hps.model.gen_istft_n_fft).to(device)
        self.device = device
        return self

    def get_arpa_text(self, text):
        text_pre = arpa_cleaners(text)
        text_norm = cleaned_text_to_sequence(text_pre)
        if self.hps.data.add_blank:
            text_norm = mathutils.intersperse(text_norm, 0)
        torch = require_torch("VITS inference")
        return torch.LongTensor(text_norm)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = mathutils.intersperse(text_norm, 0)
        torch = require_torch("VITS inference")
        return torch.LongTensor(text_norm)

    def infer(self, in_text, in_em="", in_speed=1.0, max_len=None):
        """Synthesize ``in_text``; returns float32 numpy audio."""
        torch = require_torch("VITS inference")
        if "arpa_cleaners" in self.hps.data.text_cleaners:
            stn_tst = self.get_arpa_text(in_text)
        else:
            stn_tst = self.get_text(in_text)
        if len(in_em) == 0:
            in_em = in_text
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            len_scale = torch.FloatTensor([in_speed]).to(self.device)
            em = self.moji(in_em).squeeze().unsqueeze(0).to(self.device)
            bert_emb, _ = self.bert_f.infer(in_text)
            bert_emb = bert_emb.to(self.device)
            bert_lens = torch.LongTensor([bert_emb.size(1)]).to(self.device)
            spec, phase, x1, attn = self.model.infer_ts_noistft(
                x_tst, x_tst_lengths, em, bert_emb, bert_lens, len_scale)
            audio = self.stft.inverse(spec, phase)
            audio = audio.squeeze().cpu().data.float().numpy()
        return np.asarray(audio, dtype=np.float32)

    @property
    def sampling_rate(self):
        return self.hps.data.sampling_rate


def export_torchscript(checkpoint, config_path, out_path, device="cpu",
                       test_string="Hello world", bert_model_name=None):
    """Export a trained checkpoint to TorchScript (upstream ``export_ts.py``).

    Builds the model, loads the checkpoint, runs one inference through
    ``infer_ts`` to warm the trace, then traces ``infer_ts_noistft``.
    """
    torch = require_torch("exporting a VITS model to TorchScript")
    from .bertfe import DEFAULT_MODEL, BERTFrontEnd
    from .models import SynthesizerTrn
    from .moji import TorchMoji
    from .text.symbols import symbols

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    hps = load_config(config_path)
    moji = TorchMoji()
    bert_f = BERTFrontEnd("cuda" in device,
                          model_name=bert_model_name or DEFAULT_MODEL)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_g.eval()
    train_utils.load_checkpoint(checkpoint, net_g, None)

    if "arpa_cleaners" in hps.data.text_cleaners:
        text_pre = arpa_cleaners(test_string)
        text_norm = cleaned_text_to_sequence(text_pre)
    else:
        text_norm = text_to_sequence(test_string, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = mathutils.intersperse(text_norm, 0)
    x_tst = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([len(text_norm)]).to(device)
    len_scale = torch.FloatTensor([1.0]).to(device)
    em = moji(test_string).squeeze().unsqueeze(0).to(device)
    bert_emb, _ = bert_f.infer(test_string)
    bert_emb = bert_emb.to(device)
    bert_lens = torch.LongTensor([bert_emb.size(1)]).to(device)

    net_g.infer_ts(x_tst, x_tst_lengths, em, bert_emb, bert_lens, len_scale)
    traced = torch.jit.trace_module(
        net_g,
        {"infer_ts_noistft": (x_tst, x_tst_lengths, em, bert_emb, bert_lens, len_scale)},
        check_trace=False, strict=True)
    torch.jit.save(traced, out_path)
    return out_path
