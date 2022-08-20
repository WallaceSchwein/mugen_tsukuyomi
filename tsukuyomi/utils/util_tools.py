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
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "tsukuyomi/config.yml")
CONFIG_PATH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath("__file__")), "scripts/config.yml")

def load_config(config_path: str):
    """Loads the service's config file in any file."""

    with open(config_path, 'r') as yml:
        return yaml.load(yml, Loader=yaml.FullLoader)
