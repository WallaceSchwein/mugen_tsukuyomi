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

from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
from typing import List, Tuple, Union

from . import features
from tsukuyomi.business_layer.utils import CustomMixin
from tsukuyomi.presentation_layer.data_model import Item
from tsukuyomi.utils.util_tools import CONFIG_PATH_SCRIPT, load_config

script_config = load_config(CONFIG_PATH_SCRIPT)


class TsukuyomiFeatures(CustomMixin, TransformerMixin):

    __name__ = "TsukuyomiFeatures"
    __version__ = "1.0.0"

    FEATURES = {"GeneralCharacteristicsFeatures": {},
                "POSTagFeatures": {},
                "SentimentScoreFeature": {},
                "NERFeatures": {},
                "WordVectorFeatures": {},
                "TopicAnalysisFeatures": {},
                "TFIDFVectorFeatures": {}}

    cores_involved: int = script_config['PERFORMANCE']['CORES_INVOLVED']
    feature_generators_: List[Tuple[str, Union[CustomMixin, TransformerMixin]]] = []
    no_features: List[Tuple[str, int]] = []
    exec_fit: bool = False

    def fit(self, X: List[Item], y=None, **fit_params): # NOSONAR
        self.exec_fit = True
        return self

    def transform(self, X: List[Item]): # NOSONAR
        for feat_name, feat_params in self.FEATURES.items():
            feature = getattr(features, feat_name)(**feat_params)
            self.feature_generators_.append((feat_name, feature))

        union = FeatureUnion(self.feature_generators_, n_jobs=self.cores_involved)

        if self.exec_fit:
            feature_data = union.fit_transform(X)
        else:
            feature_data = union.transform(X)
            feature_data = feature_data.reshape(1, -1)

        return feature_data
