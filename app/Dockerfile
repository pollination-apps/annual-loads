FROM ladybugtools/honeybee-energy:1.91.34 as base

WORKDIR /home/ladybugbot/app
COPY . .

USER root

RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 curl unzip -y \
    && pip3 install -r requirements.txt || echo no requirements.txt file \
    && chown -R ladybugbot /home/ladybugbot/app

USER ladybugbot
