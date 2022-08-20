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

"""setup file for project"""

from setuptools import setup, find_packages

setup_config = {"name": "tsukuyomi",
                "version": "1.0.0",
                "description": "AI-microservice for processing news articles",
                "url": "PLACEHOLDER",
                "author": "Willi Kristen",
                "author_mail": "willi.kristen@live.de",
                "license": "Willi Kristen Proprietary"}

install_requires = ["anyio==3.6.1",
                    "blis==0.7.8",
                    "catalogue==2.0.7",
                    "certifi==2022.6.15",
                    "charset-normalizer==2.1.0",
                    "click==8.1.3",
                    "cymem==2.0.6",
                    "de-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.3.0/de_core_news_lg-3.3.0-py3-none-any.whl",
                    "fastapi==0.78.0",
                    "h11==0.13.0",
                    "HanTa==0.2.0",
                    "idna==3.3",
                    "Jinja2==3.1.2",
                    "joblib==1.1.0",
                    "langcodes==3.3.0",
                    "MarkupSafe==2.1.1",
                    "murmurhash==1.0.7",
                    "nltk==3.7",
                    "numpy==1.23.1",
                    "packaging==21.3",
                    "pathy==0.6.2",
                    "preshed==3.0.6",
                    "pydantic==1.8.2",
                    "pyparsing==3.0.9",
                    "PyYAML==6.0",
                    "regex==2022.7.9",
                    "requests==2.28.1",
                    "scikit-learn==1.1.1",
                    "scipy==1.8.1",
                    "sklearn==0.0",
                    "smart-open==5.2.1",
                    "sniffio==1.2.0",
                    "spacy==3.3.1",
                    "spacy-legacy==3.0.9",
                    "spacy-loggers==1.0.2",
                    "srsly==2.4.3",
                    "starlette==0.19.1",
                    "textblob==0.17.1",
                    "textblob-de==0.4.3",
                    "thinc==8.0.17",
                    "threadpoolctl==3.1.0",
                    "tqdm==4.64.0",
                    "typer==0.4.2",
                    "typing_extensions==4.3.0",
                    "urllib3==1.26.10",
                    "uvicorn==0.18.2",
                    "wasabi==0.9.1"]

script_requires = ["bs4==0.0.1",
                   "pandas==1.4.3",
                   "tqdm-batch==0.1.0",
                   "wget==3.2"]

tests_require = ["pytest==7.1.2",
                 "testcontainers==3.6.1"]

setup_config['install_requires'] = install_requires
setup_config['script_requires'] = script_requires
setup_config['tests_require'] = tests_require
setup_config['extras_require'] = {"script": script_requires, "tests": tests_require}
setup_config['packages'] = find_packages(include=["tsukuyomi", "tsukuyomi.*"])

if __name__ == '__main__':
    setup(**setup_config)
