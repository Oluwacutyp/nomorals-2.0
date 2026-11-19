"""ARPA phonemizer: CMUdict lookup with a g2p_en fallback.

Faithful port of upstream ``text/arpa.py`` with two changes:

* The dictionary is vendored at ``nomorals/voice/vits/data/zdict.txt``
  (upstream downloaded it from Google Drive at import time when absent).
  No network access happens here; a missing vendored dict is an honest
  ``FileNotFoundError`` naming the expected path and the upstream URL.
* ``g2p_en`` is imported lazily, only when a word misses the dictionary.
  Without it, an out-of-dictionary word raises an honest error naming the
  word and the ``g2p-en`` package instead of failing at import time.
"""

import os

_DICT_URL = "https://drive.google.com/file/d/1lYKcml_bkTiOke2TMF3F_fn7UWTYHNWp/view?usp=share_link"


def default_dict_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "zdict.txt")


class ARPAPhonemizer:
    def __init__(self, dict_override=""):
        dict_fn = dict_override or default_dict_path()
        if not os.path.isfile(dict_fn):
            raise FileNotFoundError(
                f"ARPA dictionary not found at {dict_fn} (upstream downloads it from {_DICT_URL})"
            )
        self.arpa_dict = self._load_dict(dict_fn)
        self._g2p = None

    def _load_dict(self, in_dpath):
        dict_ret = {}
        with open(in_dpath, encoding="utf-8-sig") as f:
            d_lines = f.readlines()
        last_word = ""
        for d_li in d_lines:
            splits = d_li.strip().split("\t")
            word = splits[0]
            pron = splits[1]
            # only accept 1st entry for each word
            if word == last_word:
                continue
            dict_ret[word] = pron
            last_word = word
        return dict_ret

    def _g2pfil(self, intxt):
        if self._g2p is None:
            try:
                from g2p_en import G2p
            except ImportError as exc:
                raise ImportError(
                    f"word {intxt!r} is not in the ARPA dictionary and g2p-en is not "
                    "installed; install g2p-en or add the word to the dictionary"
                ) from exc
            self._g2p = G2p()
        sta = self._g2p(intxt)
        ret = " ".join(sta)
        return ret

    def doARPA(self, text):
        if text[0] == '{':
            return text
        out = ''
        for word_ in text.split(" "):
            word = word_
            end_chars = ''
            while any(elem in word for elem in r"!?,.;") and len(word) > 1:
                if word[-1] == '!':
                    end_chars = '!' + end_chars
                    word = word[:-1]
                if word[-1] == '?':
                    end_chars = '?' + end_chars
                    word = word[:-1]
                if word[-1] == ',':
                    end_chars = ',' + end_chars
                    word = word[:-1]
                if word[-1] == '.':
                    end_chars = '.' + end_chars
                    word = word[:-1]
                if word[-1] == ';':
                    end_chars = ';' + end_chars
                    word = word[:-1]
                else:
                    break
            try:
                word_arpa = self.arpa_dict[word.lower()]
            except KeyError:
                word_arpa = self._g2pfil(word)
            if len(word_arpa) != 0:
                word = "{" + str(word_arpa) + "}"
            out = (out + " " + word + end_chars).strip()
        return out
