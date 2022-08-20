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

import copy
import numpy as np
import os
import pickle
import regex
import spacy

from sklearn.base import TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from textblob_de import TextBlobDE as TBDE
from typing import List


from ..utils import CustomMixin, PATTERN_GENDER, PATTERN_SPECIAL_CHARS
from tsukuyomi.presentation_layer.data_model import Item
from tsukuyomi.utils.logging_utils import init_loggers
from tsukuyomi.utils.util_tools import CONFIG_PATH, CONFIG_PATH_SCRIPT, load_config

log = init_loggers()

script_config = load_config(CONFIG_PATH_SCRIPT)
service_config = load_config(CONFIG_PATH)

MODELS_PATH: str = os.path.join(os.path.dirname(os.path.abspath("__file__")), service_config['PATHS']['MODELS'])


class GeneralCharacteristicsFeatures(CustomMixin, TransformerMixin):
    __name__ = "GeneralCharacteristicsFeatures"
    __version__ = "1.0.0"

    pattern_gender: str = PATTERN_GENDER
    pattern_special_chars: str = PATTERN_SPECIAL_CHARS
    no_features: int = 4

    feat: np.ndarray = np.empty((352, 4))

    def fit(self, X: List[Item], y=None): # NOSONAR
        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), self.no_features))

        for idx, elem in enumerate(X):
            corpus = elem.text

            corpus = regex.sub(regex.compile('\s+'), " ", corpus).lower().strip()
            corpus = regex.sub("feind*innen", "feinde", corpus)
            corpus = regex.sub("ärzt*innen", "ärzte", corpus)
            corpus = regex.sub("akteur*innen", "akteure", corpus)
            corpus = regex.sub(regex.compile('rom_nja|rom/nja|rom:nja|rom*nja'), "roma", corpus)
            corpus = regex.sub(self.pattern_gender, "", corpus)

            corpus = regex.sub(self.pattern_special_chars, " ", corpus)
            corpus = regex.sub("cum ex", "cum/ex", corpus)
            corpus = regex.sub("cum cum", "cum/cum", corpus)
            corpus = regex.sub("biontech pfizer", "biontech/pfizer", corpus)
            corpus = regex.sub("ost west", "ost/west", corpus)
            corpus = regex.sub("mann frau", "mann/frau", corpus)
            corpus = regex.sub("junge mädchen", "junge/mädchen", corpus)
            corpus = regex.sub("km h grenze", "km/h-grenze", corpus)
            corpus = regex.sub("km h", "km/h", corpus)
            for token in regex.findall(regex.compile('^e \w+'), corpus):
                new_token = regex.sub("e ", "e-", token)
                corpus = regex.sub(token, new_token, corpus)
            corpus = regex.sub(regex.compile('[\s+]'), " ", corpus)

            sentences = 0
            words = 0
            chars = 0
            for sent in regex.split(regex.compile("[.!?:]"), corpus):
                sentences += 1
                for word in sent.split():
                    words += 1
                    for _ in word:
                        chars += 1

            tokens = 0
            letters = 0
            for token in elem.clean.split():
                tokens += 1
                for _ in token:
                    letters += 1

            self.feat[idx] = np.array([(words / sentences), (chars / words), (letters / tokens), ((words - tokens) / words * 100)])

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class POSTagFeatures(CustomMixin, TransformerMixin):
    __name__ = "POSTagFeatures"
    __version__ = "1.0.0"

    nlp: spacy.Language = spacy.load('de_core_news_lg')
    tags: dict = pickle.load(open(f"{MODELS_PATH}pos_tags.pkl", 'rb'))
    no_features: int = len(tags)

    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        self.tags = {}
        for elem in X:
            doc = self.nlp(elem.text)

            for token in doc:
                tag = token.tag_
                if tag not in self.tags:
                    self.tags[tag] = 0

        with open(f"{MODELS_PATH}pos_tags.pkl", 'wb') as p:
            pickle.dump(self.tags, p)

        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), len(self.tags)))

        for idx, elem in enumerate(X):
            tags = copy.deepcopy(self.tags)
            corpus = elem.text
            doc = self.nlp(corpus)

            for token in doc:
                tag = token.tag_
                if tag in tags:
                    tags[tag] += 1

            self.feat[idx] = np.array([v for _, v in tags.items()])

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class SentimentScoreFeature(CustomMixin, TransformerMixin):
    __name__ = "SentimentScoreFeature"
    __version__ = "1.0.0"

    no_features: int = 1

    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), 1))

        for idx, elem in enumerate(X):
            corpus = elem.clean

            blob = TBDE(corpus)

            sentiment = []
            for sentence in blob.sentences:
                sentiment.append(sentence.sentiment.polarity)

            self.feat[idx] = np.array((sum(sentiment) / len(sentiment)))

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class NERFeatures(CustomMixin, TransformerMixin):
    __name__ = "NERFeatures"
    __version__ = "1.0.0"

    nlp: spacy.Language = spacy.load('de_core_news_lg')
    labels: dict = pickle.load(open(f"{MODELS_PATH}ner_labels.pkl", 'rb'))
    no_features: int = len(labels)

    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        self.labels = {}

        for elem in X:
            doc = self.nlp(elem.text)

            for ent in doc.ents:
                if ent.label_ not in self.labels:
                    self.labels[ent.label_] = 0

        with open(f"{MODELS_PATH}ner_labels.pkl", 'wb') as p:
            pickle.dump(self.labels, p)

        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), len(self.labels)))

        for idx, elem in enumerate(X):
            labels = copy.deepcopy(self.labels)
            corpus = elem.clean
            doc = self.nlp(corpus)

            for ent in doc.ents:
                label = ent.label_
                if label in labels:
                    labels[label] += 1

            self.feat[idx] = np.array([v for _, v in labels.items()])

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class WordVectorFeatures(CustomMixin, TransformerMixin):
    __name__ = "WordVectorFeatures"
    __version__ = "1.0.0"

    nlp: spacy.Language = spacy.load('de_core_news_lg')
    no_features: int = 300

    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), self.nlp(X[0].clean).vector.shape[-1]))

        for idx, elem in enumerate(X):
            corpus = elem.clean
            self.feat[idx] = np.array(self.nlp(corpus).vector)

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class TopicAnalysisFeatures(CustomMixin, TransformerMixin):
    __name__ = "TopicAnalysisFeatures"
    __version__ = "1.0.0"

    c_vec: CountVectorizer = pickle.load(open(f"{MODELS_PATH}c_vec.pkl", 'rb'))
    lda: LDA = pickle.load(open(f"{MODELS_PATH}lda_model.pkl", 'rb'))
    no_features: int = lda.components_.shape[0]
    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        cores_involved: int = script_config['PERFORMANCE']['CORES_INVOLVED']
        max_df: float = script_config['FEATURES']['TOPIC_MODEL']['MAX_DF']
        min_df: float = script_config['FEATURES']['TOPIC_MODEL']['MIN_DF']
        cv_topic = script_config['FEATURES']['TOPIC_MODEL']['CV_TOPIC']
        params_first_iter: dict = script_config['FEATURES']['TOPIC_MODEL']['PARAMS']['FIRST_ITER']
        setup_second_iter = script_config['FEATURES']['TOPIC_MODEL']['PARAMS']['SECOND_ITER']

        self.c_vec = CountVectorizer(max_df=max_df, min_df=min_df)
        dtm = self.c_vec.fit_transform([elem.clean for elem in X])
        with open(f"{MODELS_PATH}c_vec.pkl", 'wb') as p:
                pickle.dump(self.c_vec, p)

        print("\t*******************************************\n")
        print("\tTuning for Topic Analysis started!\n\t\tBe prepared! This may take while...\n")
        print("\t*******************************************\n")

        model = LDA(random_state=99)

        print(f"\t\t[INFO] Topic Model - First iteration of tuning starts..\n")
        gridsearch = GridSearchCV(model, param_grid=params_first_iter, n_jobs=cores_involved, cv=cv_topic)
        gridsearch.fit(dtm)

        params_second_iter = {}
        for k, v in gridsearch.best_params_.items():
            match k:
                case "learning_decay":
                    threshold = setup_second_iter['learning_decay'][0]
                    step_width = setup_second_iter['learning_decay'][1]
                    if v == .99:
                        params_second_iter[k] = [.99]
                    else:
                        params_second_iter[k] = list(np.arange((v - threshold),
                                                               (v + threshold + step_width),
                                                               step_width))
                case "max_iter":
                    threshold = setup_second_iter['max_iter'][0]
                    step_width = setup_second_iter['max_iter'][1]
                    no_vals = setup_second_iter['max_iter'][2]
                    if v == params_first_iter['max_iter'][-1]:
                        params_second_iter[k] = list(np.arange(v, (v + no_vals*2*step_width), (2*step_width)))
                    else:
                        params_second_iter[k] = list(np.arange((v - threshold),
                                                               (v + threshold + step_width), 
                                                               step_width))
                case _:
                    params_second_iter[k] = [v]

        print(f"\t\t[INFO] Topic Model - Second iteration of tuning starts..\n")
        gridsearch = GridSearchCV(model, param_grid=params_second_iter, n_jobs=cores_involved, cv=cv_topic)
        gridsearch.fit(dtm)

        self.lda = gridsearch.best_estimator_
        self.lda.fit(dtm)
        with open(f"{MODELS_PATH}lda_model.pkl", 'wb') as p:
            pickle.dump(self.lda, p)

        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...")
        print(f"\t\t\tFinal Model Parameter:")
        for k, v in gridsearch.best_params_.items():
            print(f"\t\t\t\t{k}:\t{v}")
        print()

        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), self.no_features))

        for idx, elem in enumerate(X):
            corpus = elem.clean

            dtm = self.c_vec.transform([corpus])
            self.feat[idx] = self.lda.transform(dtm)[0] # topic_modelling_results

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat


class TFIDFVectorFeatures(CustomMixin, TransformerMixin):
    __name__ = "TFIDFVectorFeatures"
    __version__ = "1.0.0"

    tfidf_vec: TfidfVectorizer = pickle.load(open(f"{MODELS_PATH}tfidf_vec.pkl", 'rb'))
    pca: PCA = pickle.load(open(f"{MODELS_PATH}pca_model.pkl", 'rb'))
    no_features: int = pca.components_.shape[0]

    feat: np.ndarray

    def fit(self, X: List[Item], y=None): # NOSONAR
        max_df: float = script_config['FEATURES']['TFIDF']['MAX_DF']
        min_df: float = script_config['FEATURES']['TFIDF']['MIN_DF']
        info_cut: float = script_config['FEATURES']['TFIDF']['INFO_CUT']
        step_width: int = script_config['FEATURES']['TFIDF']['STEP_WIDTH']

        self.tfidf_vec = TfidfVectorizer(max_df=max_df, min_df=min_df) # max_features instead of PCA() ?
        tfidf_dtm = self.tfidf_vec.fit_transform([elem.clean for elem in X]).toarray()
        with open(f"{MODELS_PATH}tfidf_vec.pkl", 'wb') as p:
                pickle.dump(self.tfidf_vec, p)

        pca = PCA()
        pca.fit(tfidf_dtm)
        cut = tfidf_dtm.shape[1]
        while np.sum(self.pca.explained_variance_ratio_[:cut]) > info_cut:
            cut -= step_width
        cut += step_width

        self.pca = PCA(n_components=352)
        self.pca.fit(tfidf_dtm)
        with open(f"{MODELS_PATH}pca_model.pkl", 'wb') as p:
            pickle.dump(self.pca, p)

        print(f"\t\t[INFO] {self.__name__} set up! Features are calculated...\n")
        return self

    def transform(self, X: List[Item]): # NOSONAR
        self.feat = np.empty((len(X), self.no_features))

        for idx, elem in enumerate(X):
            corpus = elem.clean
            tfidf_dtm = self.tfidf_vec.transform([corpus]).toarray()
            self.feat[idx] = self.pca.transform(tfidf_dtm)[0]

        print(f"Feature: {self.__name__}: {self.feat.shape}")
        return self.feat
