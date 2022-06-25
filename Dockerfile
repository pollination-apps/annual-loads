FROM python:3.7-slim as base

# Set the input arguments and environment variables
ARG OPENSTUDIO_VERSION
ARG OPENSTUDIO_FILENAME

ENV HOME_PATH='/home/ladybugbot'
ENV LBT_PATH="${HOME_PATH}/ladybug_tools"
ENV LOCAL_ENERGYPLUS_PATH="${LBT_PATH}/energyplus"

# install dependencies
RUN apt-get update \
    && apt-get -y install ffmpeg libsm6 libxext6 xvfb --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser ladybugbot --uid 1000 --disabled-password --gecos ""
USER ladybugbot
WORKDIR ${HOME_PATH}
RUN mkdir -p ${LOCAL_ENERGYPLUS_PATH}

# Expects an untarred OpenStudio download in the build context to setup EnergyPlus
COPY ${OPENSTUDIO_FILENAME}/usr/local/openstudio-${OPENSTUDIO_VERSION}/EnergyPlus \
    ${LOCAL_ENERGYPLUS_PATH}

# install the core python libraries
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt || echo no requirements.txt file