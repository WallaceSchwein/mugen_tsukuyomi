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
import os
import pytest

from tsukuyomi.presentation_layer.data_model import ScoringItem


@pytest.fixture(scope="module")
def short_preprocessed_dummy_item():
    """provides a short preprocessed dummy text"""

    SHORT_DUMMY = os.path.join(os.path.dirname(os.path.abspath("__file__")), "tests/test_data/short_feature_dummy.json")

    dummy_dict = json.load(open(SHORT_DUMMY, 'r'))
    dummy_item = ScoringItem(published=dummy_dict['published'],
                             platform=dummy_dict['platform'],
                             author=dummy_dict['author'],
                             url=dummy_dict['url'],
                             title=dummy_dict['title'],
                             text=dummy_dict['text'],
                             clean=dummy_dict['clean'])
    return dummy_item


@pytest.fixture(scope="module")
def long_preprocessed_dummy_item():
    """provides a long preprocessed dummy text"""

    LONG_DUMMY = os.path.join(os.path.dirname(os.path.abspath("__file__")), "tests/test_data/long_feature_dummy.json")

    dummy_dict = json.load(open(LONG_DUMMY, 'r'))
    dummy_item = ScoringItem(published=dummy_dict['published'],
                             platform=dummy_dict['platform'],
                             author=dummy_dict['author'],
                             url=dummy_dict['url'],
                             title=dummy_dict['title'],
                             text=dummy_dict['text'],
                             clean=dummy_dict['clean'])
    return dummy_item