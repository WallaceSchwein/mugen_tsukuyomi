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

from fastapi import FastAPI

from .presentation_layer.data_model import ScoringItem
from .presentation_layer.service import Tsukuyomi
from .utils.logging_utils import init_loggers

BASE_URL = "https://tsukuyomi.com/"
API_DESCRIPTION = "The JSON-item, you want to get scored by the service's model."

log = init_loggers()

app = FastAPI()
service = Tsukuyomi()

log.info("Service logger connected!")

@app.get('/')
async def main(): # TODO suggestion: implement base page
    return {"Welcome": "This is the API of tsukuyomi microservice"}

@app.post('/score',
          summary="Score a digital news article.",
          response_description="JSON-object, now containing the similarity rate with the political party's content.")
async def score(item: ScoringItem):
    """
    Tsukuyomi microservice will calculate the similarity of a news article and the content,
    of the established political parties.
    Args:
        item (Item): JSON-object, containing the news article and some additional meta data.
    Returns:
        item (Item): Input-object, now additionally containing the calculated similarity rate between the scored article 
                     and the content of the political parties.
    """

    log.info("Object has been received for scoring..")

    # if item['lang'] == "EN":
    #     return service.tsukuyomi_score_EN(item).json()
    # elif item['lang'] == "DE":
    #     return service.tsukuyomi_score_DE(item).json()
    # elif item['lang'] == "ES":
    #     return service.tsukuyomi_score_ES(item).json()

    return json.dumps(service.tsukuyomi_score(item).__dict__)

@app.post('/topic',
          summary="Calculate topic affiliation of a news article.",
          response_description="JSON-object, now containing the affiliation rate to the topics, discussed in the national parliament.")
async def score(item: ScoringItem):
    """
    Tsukuyomi microservice will calculate the affiliation of a news article to the dominant topics, discussed by the established 
    political parties, in the national parliament.
    Args:
        item (Item): JSON-object, containing the news article and some additional meta data.
    Returns:
        item (Item): Input-object, now additionally containing the calculated topic affiliation rate of the article, 
                     to the national parliament's most dominant political subjects.
    """

    # if item['lang'] == "EN":
    #     return service.tsukuyomi_topic_EN(item).json()
    # elif item['lang'] == "DE":
    #     return service.tsukuyomi_topic_DE(item).json()
    # elif item['lang'] == "ES":
    #     return service.tsukuyomi_topic_ES(item).json()

    log.info("Object has been received for topic modelling..")

    return json.dumps(service.tsukuyomi_topic(item).__dict__)

@app.get('/health',
         summary="Health endpoint",
         status_code=200)
async def health():
    """
    A Health endpoint, to ping for checking if service is up and running.
    Returns:
        status_code: Returns <200> if service is up and running.
    """
    return service._health()