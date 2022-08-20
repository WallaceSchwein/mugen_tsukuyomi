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
import numpy as np
import os
import pickle

from sklearn.base import BaseEstimator
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

from tsukuyomi.business_layer.utils import CustomMixin
from tsukuyomi.presentation_layer.data_model import ScoringItem
from tsukuyomi.utils.util_tools import load_config, CONFIG_PATH

config = load_config(CONFIG_PATH)

MODELS_PATH: str = os.path.join(os.path.dirname(os.path.abspath("__file__")), config['PATHS']['MODELS'])

class TsukuyomiClassifier(CustomMixin):

    __name__ = "TsukuyomiClassifier"
    __version__ = "1.0.0"

    model: BaseEstimator = pickle.load(open(f"{MODELS_PATH}service_clf.pkl", 'rb'))

    def fit(self, X: np.ndarray, y: np.ndarray): # NOSONAR
        return self

    def predict(self, X: np.ndarray): # NOSONAR
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray): # NOSONAR
        return self.model.predict_proba(X)

class TsukuyomiTopicModel(CustomMixin):

    __name__ = "TsukuyomiTopicModel"
    __version__ = "1.0.0"

    vectorizer: CountVectorizer = pickle.load(open(f"{MODELS_PATH}c_vec.pkl", 'rb'))
    model: LDA = pickle.load(open(f"{MODELS_PATH}lda_model.pkl", 'rb'))

    no_topics: int = model.components_.shape[0]

    def fit(self, X: List[ScoringItem], y: np.ndarray): # NOSONAR
        return self

    def predict(self, X: List[ScoringItem]): # NOSONAR
        return np.array([np.argmax(item) for item in self.predict_proba(X)])

    def predict_proba(self, X: List[ScoringItem]): # NOSONAR
        res = np.empty((len(X), self.no_topics))
        for idx, elem in enumerate(X):
            corpus = elem.clean

            dtm = self.vectorizer.transform([corpus])
            res[idx] = self.model.transform(dtm)[0]

        return res
