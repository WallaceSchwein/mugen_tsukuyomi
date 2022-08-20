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

import inspect
import regex

from importlib import import_module
from sklearn.base import BaseEstimator
from ..utils.exceptions import InstanceLoadingError

PATTERN_GENDER: regex.Pattern = regex.compile('\/Influencerinnen|\/innen|\:innen|\*innen|\_innen|\/in|\:in|\*in|\_in|\/zze|\:zze|\*zze|\_zze|\/ze|\:ze|\*ze|\_ze')
PATTERN_SIGNS_IN_WORDS: regex.Pattern = regex.compile('\p{Ll}\/\p{Ll}|\p{Ll}\_\p{Ll}|\p{Ll}\*\p{Ll}|\p{Ll}\:\p{Ll}|\p{Ll}\-\p{Ll}|\p{Ll}\–\p{Ll}')
PATTERN_SPECIAL_CHARS: regex.Pattern = regex.compile('[–+-/\\(){}[\]<>§$%&=:\*°#@€¿&_ʼ‘’"„“”᾽‧¸‚,;…\'~‐‑‒]')


class CustomMixin(BaseEstimator):
    """Mixin to be used, to save sklearn style estimators."""

    @classmethod
    def get_version(cls, version: str):
        """Returns an instance of a special version of the class"""
        name = cls.__name__
        module = cls.__module__
        
        try:
            module_ref = import_module(module)
            classes = {obj.__version__: obj for _, obj in inspect.getmembers(module_ref)
                                        if isinstance(obj, type) and issubclass(obj, getattr(module_ref, name))}
        except AttributeError as e:
            raise InstanceLoadingError(f"Could not load instance of class: '{name}' "
                                       f"does not exist in module: '{module}'") from e
        try:
            return classes[version]
        except KeyError as e:
            raise InstanceLoadingError(f"Could not find version: {version} for class: {name}") from e
