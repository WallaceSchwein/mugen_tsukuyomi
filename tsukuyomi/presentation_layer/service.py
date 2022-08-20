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
import pickle

from sklearn.pipeline import Pipeline

from tsukuyomi.business_layer.features.feature_generation import TsukuyomiFeatures
from tsukuyomi.business_layer.preprocessing.preprocessing import TsukuyomiTransformer
from tsukuyomi.business_layer.scoring.model import TsukuyomiClassifier, TsukuyomiTopicModel
from tsukuyomi.presentation_layer.data_model import ScoringItem
from tsukuyomi.utils.logging_utils import init_loggers
from ..utils.util_tools import load_config, CONFIG_PATH

config = load_config(CONFIG_PATH)

log = init_loggers()

MODELS_PATH: str = os.path.join(os.path.dirname(os.path.abspath("__file__")), config['PATHS']['MODELS'])


class TsukuyomiBase:

    name: str
    config: dict
    #logger: 

    def _health(self):
        info = {"status": "UP",
                "service": "tsukuyomi"}
        return 200, info


class Tsukuyomi(TsukuyomiBase):

    __name__ = "Tsukuyomi"
    __version__ = "1.0.0"

    name = 'tsukuyomi'
    config = config

    scaler = pickle.load(open(f"{MODELS_PATH}scaler.pkl", 'rb'))

    topic_pipe = Pipeline([("preprocessing_transformer", TsukuyomiTransformer()),
                           ("topic_model", TsukuyomiTopicModel())])

    scoring_pipe = Pipeline([("preprocessing_transformer", TsukuyomiTransformer()),
                             ("feature_generator", TsukuyomiFeatures()),
                             ("standard_scaler", scaler),
                             ("scoring_model", TsukuyomiClassifier())])

    def tsukuyomi_score(self, X: ScoringItem): # NOSONAR
        """
        Main service function for scoring the request's objects by applying the Tsukuyomi scoring pipeline.

        Args:
            X (ScoringItem): Received object to score, mapped to the service's data model.

        Returns:
            ScoringItem: Returns the same Item with additional data added at the key 'scoring'.
        """

        probas = self.scoring_pipe.predict_proba([X])

        X.score = {}
        for idx, proba in enumerate(probas[0]):
            match idx:
                case 0:
                    X.score['CDU/CSU'] = float(proba)
                case 1:
                    X.score['SPD'] = float(proba)
                case 2:
                    X.score['AfD'] = float(proba)
                case 3:
                    X.score['FDP'] = float(proba)
                case 4:
                    X.score['BÜNDNIS 90/DIE GRÜNEN'] = float(proba)
                case 5:
                    X.score['DIE LINKE'] = float(proba)

        log.info("Object has been scored succesfully!")

        return X

    def tsukuyomi_topic(self, X: ScoringItem): # NOSONAR
        """
        Main service function for calculate the request's object's topics 
        by applying the Tsukuyomi topic modelling pipeline.

        Args:
            X (ScoringItem): Received object to topic score, mapped to the service's data model.

        Returns:
            ScoringItem: Returns the same Item with additional data added at the key 'topic'.
        """

        probas = self.topic_pipe.predict_proba([X])

        X.topic = {}
        for idx, proba in enumerate(probas[0]):
            idx += 1
            name = f"Topic_no_{idx}" if config['TOPICS'][f'TOPIC_NO_{idx}'] == "" else config['TOPICS'][f'TOPIC_NO_{idx}']

            X.topic[name] = float(proba)

        log.info("The topics for this object have been calculated succesfully!")

        return X
