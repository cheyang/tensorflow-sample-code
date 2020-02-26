# docker build -t registry.cn-shanghai.aliyuncs.com/tensorflow-samples/tensorflow-gpu-mem:10.0-runtime-centos7 .
FROM registry.cn-huhehaote.aliyuncs.com/tensorflow-samples/tensorflow:centos7-cuda10.0-1.14-py36

ADD main.py /app/main.py

CMD ["python3","/app/main.py"]
