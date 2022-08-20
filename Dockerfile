# Dockerfile for tsukuyomi microservice
# The python implementation of helpful and somehow generic functions

# Created: April 2022
# @author: Willi Kristen

# @license: Willi Kristen:
# Copyright (c) 2022 Willi Kristen, Germany
# https://de.linkedin.com/in/willi-kristen-406887218

# All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution.
# This software is the confidential and proprietary information of Willi Kristen.
# You shall not disclose such confidential information and shall use it only in accordance with the
# term s of the license agreement you entered into with Willi Kristen's software solutions.

FROM python:3.10

LABEL maintainer="s_kristenw@hwr-berlin.de"
LABEL version="1.0"

RUN apt update && apt install -y git
# Add here packages to install

# When running docker compose
# EXPOSE 8000

ENV PYTHON_VERSION="3.10" \
    # Keeps Python from generating .pyc files in the container
    PYTHONDONTWRITEBYTECODE=1 \
    # Turns off buffering for easier container logging
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app/

# Install pip requirements
RUN python -m pip install --upgrade pip && \
    python -m pip install .

# Install script requirements
RUN python -m pip install .[script]

# Install tests requirements
RUN python -m pip install .[tests]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 9999 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
RUN echo "export PATH='/home/appuser/.local/bin:$PATH'" >> ~/.bashrc && . ~/.bashrc

# # Change mod for init script to executable
# RUN chmod +x /app/start.sh

# # Set entry point
# ENTRYPOINT ["start.sh"]

CMD [ "uvicorn", "tsukuyomi.run:app", "--host", "0.0.0.0", "--port", "8000" ]
