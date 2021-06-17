FROM python:3.7

EXPOSE 8080

# For librosa
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev \
&& apt-get -y install ffmpeg

COPY ./requirements.txt /inference-api/requirements.txt
RUN pip install -r /inference-api/requirements.txt

RUN mkdir /inference-api/tmp

# Add crontab file to clean cached files
ADD crontab /etc/cron.d/gc-cron
RUN chmod 0644 /etc/cron.d/gc-cron
RUN apt-get -y install cron
CMD cron

COPY app/* /inference-api/app/
COPY html/* /inference-api/html/
COPY resources/* /inference-api/resources/
COPY models/* /inference-api/models/
COPY server.py /inference-api/server.py
COPY ./config.yml /inference-api/config.yml

WORKDIR /inference-api/

CMD [ "python", "/inference-api/server.py" ]