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

import pytest
from unittest import TestCase

import json
import os

import regex

from tsukuyomi.business_layer.preprocessing.preprocessing import TsukuyomiTransformer
from tsukuyomi.business_layer.preprocessing.preprocessing_steps import GenderFilter, CompoundTermFilter, StopWordFilter, Lemmatizer
from tsukuyomi.presentation_layer.data_model import ScoringItem

tc = TestCase()


class TestPreprocessing:
    """Unit tests for preprocessing based on easy dummy data operations."""

    DUMMY_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "tests/test_data/preprocessing_dummy.json")

    @pytest.fixture(scope="class")
    def dummy_item(self):
        """ Provides a input dummy item."""

        dummy_dict = json.load(open(self.DUMMY_PATH, 'r'))
        dummy_item = ScoringItem(published=dummy_dict['published'],
                                 platform=dummy_dict['platform'],
                                 author=dummy_dict['author'],
                                 url=dummy_dict['url'],
                                 title=dummy_dict['title'],
                                 text=dummy_dict['text'])
        return dummy_item

    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_preprocessing_transformer(self, version, dummy_item):
        """Unit Test for the preprocessing transformer"""

        RESULT = "ergebnis wissenschaftlich langzeit studi arzt begleiten eindeutig teilnehmer einflussen politik aussetzen zeigen deutlich veränderung präfrontalen kortizes mann frau vergleich legen geschlecht prozeß einflussen"

        transformer_class = TsukuyomiTransformer.get_version(version=version)
        transformer = transformer_class()

        dummy_transformed = transformer.transform([dummy_item])

        tc.assertEqual(dummy_transformed[0].clean, RESULT)

    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_gender_filter(self, version, dummy_item):
        """Unit Test for the gender filter"""

        RESULT = "die ergebnisse wissenschaftlicher langzeit-studien, welche von ärzte begleitet wurden, sind eindeutig! teilnehmer, die über mehrere jahre dem einfluss von politik ausgesetzt waren, zeigten deutliche veränderungen der präfrontalen kortizes. der mann/frau-vergleich legt nahe, dass das geschlecht auf diesen prozess keinen einfluss hat.."

        gender_filter_class = GenderFilter.get_version(version=version)
        gender_filter = gender_filter_class()

        corpus = dummy_item.text.lower().strip().split()

        corpus_transformed = " ".join(gender_filter.fit_transform(corpus))

        tc.assertEqual(corpus_transformed, RESULT)

    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_compound_noun_filter(self, version, dummy_item):
        """Unit Test for the compound noun filter"""

        RESULT = "die ergebnisse wissenschaftlicher langzeit studien  welche von ärzt innen begleitet wurden  sind eindeutig! teilnehmer innen  die über mehrere jahre dem einfluss von politik ausgesetzt waren  zeigten deutliche veränderungen der präfrontalen kortizes  der mann frau vergleich legt nahe  dass das geschlecht auf diesen prozess keinen einfluss hat  "

        compound_noun_filter_class = CompoundTermFilter.get_version(version=version)
        compound_noun_filter = compound_noun_filter_class()

        corpus = dummy_item.text.lower().strip().split()

        # no fit, to not modify 'signs_in_words.txt'
        corpus_transformed = " ".join(compound_noun_filter.transform(corpus))

        tc.assertEqual(corpus_transformed, RESULT)

    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_stop_word_filter(self, version, dummy_item):
        """Unit Test for the stop word filter"""

        RESULT = "ergebnisse wissenschaftlicher langzeit-studien, ärzt:innen begleitet wurden, eindeutig! teilnehmer:innen, einfluss politik ausgesetzt waren, zeigten deutliche veränderungen präfrontalen kortizes. mann/frau-vergleich legt nahe, geschlecht prozess einfluss hat.."

        stop_word_filter_class = StopWordFilter.get_version(version=version)
        stop_word_filter = stop_word_filter_class()

        corpus = dummy_item.text.lower().strip().split()

        corpus_transformed = " ".join(stop_word_filter.fit_transform(corpus))

        tc.assertEqual(corpus_transformed, RESULT)

    @pytest.mark.unit
    @pytest.mark.preprocessing
    @pytest.mark.parametrize('version', ['1.0.0'])
    def test_lemmatizer(self, version, dummy_item):
        """Unit Test for the stop word filter"""

        RESULT = "die ergebnis wissenschaftlich langzeit-studien, welche von ärzt:innen begleiten wurden, sein eindeutig! teilnehmer:innen, die über mehrere jahr dem einflussen von politik aussetzen waren, zeigen deutlich veränderung der präfrontalen kortizes. der mann/frau-vergleich legen nahe, dass das geschlecht auf diesen prozeß keinen einflussen hat.."

        lemmatizer_class = Lemmatizer.get_version(version=version)
        lemmatizer = lemmatizer_class()

        corpus = dummy_item.text.lower().strip().split()

        corpus_transformed = " ".join(lemmatizer.fit_transform(corpus))

        tc.assertEqual(corpus_transformed, RESULT)