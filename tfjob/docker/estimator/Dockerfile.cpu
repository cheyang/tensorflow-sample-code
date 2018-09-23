FROM tensorflow/tensorflow:1.10.1-py3

RUN mkdir -p /app/MNIST/ && \
    cd  /app/MNIST/ && \
    curl -O http://kubeflow-oss.oss-cn-hangzhou.aliyuncs.com/tensorflow/input_data/t10k-images-idx3-ubyte.gz && \
    curl -O http://kubeflow-oss.oss-cn-hangzhou.aliyuncs.com/tensorflow/input_data/t10k-labels-idx1-ubyte.gz && \
    curl -O http://kubeflow-oss.oss-cn-hangzhou.aliyuncs.com/tensorflow/input_data/train-images-idx3-ubyte.gz && \
    curl -O http://kubeflow-oss.oss-cn-hangzhou.aliyuncs.com/tensorflow/input_data/train-labels-idx1-ubyte.gz

RUN sed -i 's/https:\/\/storage.googleapis.com\/cvdf-datasets\/mnist\//http:\/\/kubeflow-oss.oss-cn-hangzhou.aliyuncs.com\/tensorflow\/input_data\//g' /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py

COPY mnist_estimator.py /app/mnist_estimator.py

ENTRYPOINT ["python", "/app/mnist_estimator.py"]