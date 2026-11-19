# imports
import json
import os

import torch

from .global_variables import PRETRAINED_PATH, VOCAB_PATH
from .model_def import torchmoji_feature_encoding
from .sentence_tokenizer import SentenceTokenizer


class TorchMoji:
    def __init__(self, verbose=False):
        if not os.path.exists(PRETRAINED_PATH):
            raise FileNotFoundError(
                f"TorchMoji weights not found at {PRETRAINED_PATH}. They are not shipped in "
                "the vits-finetuning clone; place pytorch_model.bin there "
                "(from the torchMoji release) to enable the emotion frontend."
            )
        if verbose:
            print(f'Tokenizing using dictionary from {VOCAB_PATH}')
        with open(VOCAB_PATH) as f:
            vocabulary = json.load(f)

        self.torchmoji_tokenizer = SentenceTokenizer(vocabulary, fixed_length=120)

        if verbose:
            print(f'Loading model from {PRETRAINED_PATH}.')
        self.torchmoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)

    def __call__(self, text):# str[txt_T]
        with torch.no_grad():
            tokenized, _, _ = self.torchmoji_tokenizer.tokenize_sentences([text,])
            embed = self.torchmoji_model(tokenized)
        return torch.from_numpy(embed).unsqueeze(0).float()# [1, 1, embed]
