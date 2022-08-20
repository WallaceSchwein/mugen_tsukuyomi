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

import json
import numpy as np
import pytest

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Type, Union, Optional
from unittest import TestCase

from tsukuyomi.business_layer.features import features, feature_generation
from tsukuyomi.presentation_layer.data_model import ScoringItem

tc = TestCase()

def feature_transformer_tests(feat_gen_cls: Union[Type[TransformerMixin], Type[BaseEstimator]],
                              n_features_out: int,
                              data_in: List[ScoringItem],
                              data_out: Optional[np.ndarray] = None):
    """Collection of basic tests to ensure Transformers behave like intended"""

    feat_gen = feat_gen_cls()
    transformed = feat_gen.transform(data_in)

    tc.assertEquals(n_features_out, transformed.shape[1])

    if data_out is not None:
        np.testing.assert_array_almost_equal(data_out, transformed)


@pytest.mark.usefixtures('short_preprocessed_dummy_item',
                         'long_preprocessed_dummy_item')
class TestFeatureGeneration:
    """Unit tests for feature generation based on easy dummy data operations."""

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_general_characteristics_feature(self,
                                             short_preprocessed_dummy_item,
                                             long_preprocessed_dummy_item):
        """Unit test for General Characteristics Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.GeneralCharacteristicsFeatures,
                                  features.GeneralCharacteristicsFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.GeneralCharacteristicsFeatures,
                                  features.GeneralCharacteristicsFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_POStag_feature(self,
                            short_preprocessed_dummy_item,
                            long_preprocessed_dummy_item):
        """Unit test for POS-Tag Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.POSTagFeatures,
                                  features.POSTagFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.POSTagFeatures,
                                  features.POSTagFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_sentiment_score_feature(self,
                                     short_preprocessed_dummy_item,
                                     long_preprocessed_dummy_item):
        """Unit test for Sentiment Score Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.SentimentScoreFeature,
                                  features.SentimentScoreFeature.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.SentimentScoreFeature,
                                  features.SentimentScoreFeature.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_NER_feature(self,
                         short_preprocessed_dummy_item,
                         long_preprocessed_dummy_item):
        """Unit test for NER Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.NERFeatures,
                                  features.NERFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.NERFeatures,
                                  features.NERFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_word_vector_feature(self,
                                 short_preprocessed_dummy_item,
                                 long_preprocessed_dummy_item):
        """Unit test for Word Vector Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.WordVectorFeatures,
                                  features.WordVectorFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.WordVectorFeatures,
                                  features.WordVectorFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    def test_topic_analysis_feature(self,
                                    short_preprocessed_dummy_item,
                                    long_preprocessed_dummy_item):
        """Unit test for Topic Analysis Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.TopicAnalysisFeatures,
                                  features.TopicAnalysisFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.TopicAnalysisFeatures,
                                  features.TopicAnalysisFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    def test_tfidf_feature(self,
                           short_preprocessed_dummy_item,
                           long_preprocessed_dummy_item):
        """Unit test for General Characteristics Feature"""

        # Run test with short preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.TFIDFVectorFeatures,
                                  features.TFIDFVectorFeatures.no_features,
                                  [short_preprocessed_dummy_item])
        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(features.TFIDFVectorFeatures,
                                  features.TFIDFVectorFeatures.no_features,
                                  [long_preprocessed_dummy_item])

    @pytest.mark.unit
    @pytest.mark.feature_generation
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_tsukuyomi_features(self, 
                                version,
                                short_preprocessed_dummy_item,
                                long_preprocessed_dummy_item):
        """Unit test for Tsukuyomi Feature Generator"""

        tf_class = feature_generation.TsukuyomiFeatures.get_version(version=version)
        tf = tf_class()

        n_features = 0
        for feat_name, feat_params in tf.FEATURES.items():
            feature = getattr(features, feat_name)(**feat_params)
            n_features += feature.no_features

        # Run test with short preprocessed dummy text and verify to get expected output
        # feature_transformer_tests(feature_generation.TsukuyomiFeatures,
        #                           n_features,
        #                           [short_preprocessed_dummy_item]) 
        # # TODO debug info: Pipeline calls every feature 2x in second call of test function

        # Run test with long preprocessed dummy text and verify to get expected output
        feature_transformer_tests(feature_generation.TsukuyomiFeatures,
                                  n_features,
                                  [short_preprocessed_dummy_item])
