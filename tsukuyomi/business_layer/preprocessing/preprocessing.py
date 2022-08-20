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
terms of the license agreement you entered into with Willi Kristen's software solutions.
"""

from typing import List
import regex

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from tsukuyomi.presentation_layer.data_model import Item
from .preprocessing_steps import GenderFilter, CompoundTermFilter, StopWordFilter, Lemmatizer
from ..utils import CustomMixin


class TsukuyomiTransformer(CustomMixin, TransformerMixin):

    __name__ = "TsukuyomiTransformer"
    __version__ = "1.0.0"

    preprocessing_pipeline: Pipeline = Pipeline([('gender_filter', GenderFilter()),
                                                 ('compound_term_filter', CompoundTermFilter()),
                                                 ('stop_word_filter', StopWordFilter()),
                                                 ('lemmatizer', Lemmatizer())])

    def fit(self, X: List[Item], y=None): # NOSONAR
        return self

    def transform(self, X: List[Item]): # NOSONAR
        for elem in X:
            corpus = elem.text

            corpus = regex.sub(regex.compile('[!?.,;0123456789]'), "", corpus)
            corpus = regex.sub(regex.compile('\s+'), " ", corpus).lower().strip().split()

            corpus_transformed = self.preprocessing_pipeline.transform(corpus) # NOSONAR

            corpus_transformed = " ".join(corpus_transformed)
            corpus_transformed = regex.sub(regex.compile('\s+'), " ", corpus_transformed)

            elem.clean = corpus_transformed

        return X
