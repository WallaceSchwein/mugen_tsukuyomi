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

from numpy import ndarray
from pydantic import BaseModel
from typing import Optional


class Item(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    clean: Optional[str]
    score: Optional[dict]
    topic: Optional[dict]


class ScoringItem(Item): #article

    published: str
    platform: str
    author: str
    url: str
    title: str
    text: str


class TrainingItem(Item): #speech

    date: str
    meeting_no: str
    period: str
    speaker: str
    party: str
    text: str
