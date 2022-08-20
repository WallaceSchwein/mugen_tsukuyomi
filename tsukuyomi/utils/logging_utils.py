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

import logging
import uvicorn

from tsukuyomi.utils.util_tools import CONFIG_PATH, load_config

config = load_config(CONFIG_PATH)

LOGGING_CONFIG = config['LOGGING']

FORMAT = config['LOGGING']['FORMAT']

def init_loggers():
    """Initializes a logger in any file of the Tsukuyomi service"""

    logger = logging.getLogger("tsukuyomi_log")
    logger.setLevel(logging.INFO) # logging.DEBUG

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = formatter = uvicorn.logging.DefaultFormatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
