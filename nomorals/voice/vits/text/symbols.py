"""Symbol table for the VITS text front end.

Faithful port of upstream ``text/symbols.py``. Upstream builds the ARPAbet
tail from ``g2p_en``'s phoneme list at import time; this port vendors that
list as a constant (it is a fixed property of the g2p_en release) so the
symbol table — and therefore every text id — is available with no optional
dependency installed. When g2p_en *is* installed the test battery verifies
the vendored list against it.
"""

_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# The g2p_en phoneme list (74 entries), captured from g2p_en at survey time
# and stored in ~/workspace/vits-verify/g2p_phonemes.json. Upstream prefixes
# each entry with '@' to form the ARPAbet symbols.
_G2P_PHONEMES = [
    '</s>', '<pad>', '<s>', '<unk>',
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
    'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
    'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
    'V', 'W', 'Y', 'Z', 'ZH',
]
_arpabet = ["@" + s for s in _G2P_PHONEMES]

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + _arpabet

# Special symbol ids
SPACE_ID = symbols.index(" ")


def g2p_phonemes():
    """Return the vendored g2p_en phoneme list (without the '@' prefix)."""
    return list(_G2P_PHONEMES)
