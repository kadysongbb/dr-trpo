FROM tensorflow/tensorflow:latest-gpu-py3

RUN rm -rf /usr/src/app
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
USER root

RUN pip install -U setuptools
RUN pip install -U wheel
RUN pip install -U flask-cors

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . /usr/src/app

CMD ["python", "-m", "main"]
CMD tail -f /dev/null
