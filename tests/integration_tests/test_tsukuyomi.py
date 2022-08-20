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
import requests

from unittest import TestCase

tc = TestCase()


@pytest.mark.usefixtures('provide_service', 'long_preprocessed_dummy_item')
class TestTsukuyomiIntegration:
    """Integration test for the full tsukuyomi service"""

    URL = "http://127.0.0.1:5008/"

    HEALTH_SUFFIX = "health"
    SCORE_SUFFIX = "score"
    TOPIC_SUFFIX = "topic"

    headers = {"accept": "application/json",
               "Content-type": "application/json"}

    @pytest.mark.integration
    def test_scoring_endpoint(self, provide_service, long_preprocessed_dummy_item):
        """Test for the scoring endpoint of the service."""

        _ = provide_service
        dummy = long_preprocessed_dummy_item.__dict__

        res_health = requests.get(url=f"{self.URL}{self.HEALTH_SUFFIX}")
        tc.assertEqual(res_health.status_code, 200)

        res_score = requests.post(url=f"{self.URL}{self.SCORE_SUFFIX}", json=dummy, headers=self.headers)

        tc.assertEqual(res_score.status_code, 200)
        tc.assertIn("scoring", res_score.json())
        tc.assertAlmostEqual(sum([v for _, v in res_score.json()['scoring'].items()]), 100)

    @pytest.mark.integration
    def test_topic_modelling_endpoint(self, provide_service, long_preprocessed_dummy_item):
        """Test for the scoring endpoint of the service."""

        _ = provide_service
        dummy = long_preprocessed_dummy_item.__dict__

        res_health = requests.get(url=f"{self.URL}{self.HEALTH_SUFFIX}")
        tc.assertEqual(res_health.status_code, 200)

        res_top = requests.post(url=f"{self.URL}{self.TOPIC_SUFFIX}", json=dummy, headers=self.headers)

        tc.assertEqual(res_top.status_code, 200)
        tc.assertIn("topic", res_top.json())
        tc.assertAlmostEqual(sum([v for _, v in res_top.json()['topic'].items()]), 100)
