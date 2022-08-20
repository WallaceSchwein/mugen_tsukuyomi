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

from typing import Optional
import pytest

from testcontainers.core.container import DockerContainer

@pytest.fixture(scope="class")
def provide_service(service_config: Optional[dict]=None):
    """
    Provides a full service by yielding a running tsukuyomi test container for the duration
    of the execution of the test class
    """

    with DockerContainer('tsukuymi:latest') as service:
        service.start()
        try:
            yield service
        finally:
            service.stop()
