"""
tsukuyomi microservice
The python implementation of helpful and somehow generic functions

Created: April 2022
@author: Willi Kristen

@license: Willi Kristen:
Copyright (c) 2022 Willi Kristen, Germany
https://de.linkedin.com/in/willi-kristen-406887218

All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution.
This software is the confidential and proprietary information of Willi Kristen.
You shall not disclose such confidential information and shall use it only in accordance with the
term s of the license agreement you entered into with Willi Kristen's software solutions.
"""

import os
import regex

from HanTa import HanoverTagger as HT
from sklearn.base import TransformerMixin
from typing import List

from ..utils import CustomMixin, PATTERN_GENDER, PATTERN_SIGNS_IN_WORDS, PATTERN_SPECIAL_CHARS
from tsukuyomi.utils.util_tools import load_config, CONFIG_PATH

config = load_config(CONFIG_PATH)

PP_FILES_PATH: str = os.path.join(os.path.dirname(os.path.abspath("__file__")), config['PATHS']['PP_FILES_PATH'])


class GenderFilter(CustomMixin, TransformerMixin):
    __name__ = "GenderFilter"
    __version__ = "1.0.0"

    pattern_gender: regex.Pattern = PATTERN_GENDER

    gender_terms: dict[List[str], str] = {"feinde": ["feind_innen", "feind/innen", "feind:innen", "feind*innen"],
                                          "ärzte": ["ärzt_innen", "ärzt/innen", "ärzt:innen", "ärzt*innen"],
                                          "akteure": ["akteur_innen", "akteur/innen", "akteur:innen", "akteur*innen"],
                                          "roma": ["rom_nja", "rom/nja", "rom:nja", "rom*nja"]} # TODO outsource to JSON

    def fit(self, X: List[str], y=None): # NOSONAR
        return self

    def transform(self, X: List[str]):  # NOSONAR
        X_filtered = []  # NOSONAR
        for token in X:
            for replace, gender_token in self.gender_terms.items():
                if token in gender_token:
                    token = replace

            token = regex.sub(self.pattern_gender, "", token)

            X_filtered.append(token)

        return X_filtered


class CompoundTermFilter(CustomMixin, TransformerMixin):
    __name__ = "CompundTermFilter"
    __version__ = "1.0.0"

    FILE_PATH: str = f'{PP_FILES_PATH}signs_in_words.txt'
    pattern_signs_in_words: regex.Pattern = PATTERN_SIGNS_IN_WORDS
    pattern_special_chars: regex.Pattern = PATTERN_SPECIAL_CHARS

    signs_in_words: List[str] = [token for token in open(FILE_PATH, 'r').readlines()]
    compound_terms: List[str] = ["cum ex",
                                 "cum cum", 
                                 "biontech pfizer",
                                 "ost west",
                                 "mann frau",
                                 "junge mädchen",
                                 "km h grenze",
                                 "km h"]

    def fit(self, X: List[str], y=None): # NOSONAR
        return self

    def transform(self, X: List[str]): # NOSONAR
        X_filtered = [] # NOSONAR
        for token in X:
            if regex.search(self.pattern_signs_in_words, token) and token not in self.signs_in_words:
                self.signs_in_words.append(token)

            token = regex.sub(self.pattern_special_chars, " ", token)

            if token in self.compound_terms:
                token = regex.sub(" ", "/", token)
            elif regex.search(regex.compile('^e \w+'), token):
                token = regex.sub("e ", "e-", token)

            X_filtered.append(token)

        with open(self.FILE_PATH, 'w') as f:
            for term in set(self.signs_in_words):
                f.write(term + "\n")

        return X_filtered


class StopWordFilter(CustomMixin, TransformerMixin):
    __name__ = "StopWordFilter"
    __version__ = "1.0.0"

    FILTER_PATHS: List[str] = [f'{PP_FILES_PATH}representatives.txt', f'{PP_FILES_PATH}additional_terms.txt', f'{PP_FILES_PATH}stop_words_de.txt']

    filters: List[List[str]] = [[t for t in open(p, 'r').readlines()] for p in FILTER_PATHS]

    def fit(self, X: List[str], y=None): # NOSONAR
        return self

    def transform(self, X: List[str]): # NOSONAR
        X_filtered = [] # NOSONAR
        for token in X:
            for token_filter in self.filters:
                for filter_token in token_filter:
                    if filter_token.lower().strip() == token:
                        token = "" # TODO check if currently this deletes tokens unwantedly

            X_filtered.append(token)
        X_filtered = [t for t in X_filtered if t != ""]

        return X_filtered


class Lemmatizer(CustomMixin, TransformerMixin):
    __name__ = "Lemmatizer"
    __version__ = "1.0.0"

    tagger = HT.HanoverTagger('morphmodel_ger.pgz')

    def fit(self, X: List[str], y=None): # NOSONAR
        return self

    def transform(self, X: List[str]): # NOSONAR
        # return value of tag_sent: (word, lemma, pos)
        return [lemma.lower().strip() for (_, lemma, _) in self.tagger.tag_sent(X)]