FROM tensorflow/tensorflow:1.5.0

WORKDIR /app
ADD mnist_client.py /app
ADD data /app
ADD requirements.txt /app

RUN pip install -r /app/requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]