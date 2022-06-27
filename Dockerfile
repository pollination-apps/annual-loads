FROM ladybugtools/honeybee-energy:1.91.34 as base

USER root

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt || echo no requirements.txt file
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl unzip -y
