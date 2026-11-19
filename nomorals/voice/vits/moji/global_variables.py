""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend([f'{SPECIAL_PREFIX}BLANK_{i}' for i in range(6, 10)])

DATA_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = f'{DATA_PATH}/data/vocabulary.json'
PRETRAINED_PATH = f'{DATA_PATH}/data/pytorch_model.bin'

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']
