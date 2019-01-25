DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: gpu

gpu: 
	$(DOCKER) build --no-cache -f Dockerfile.gpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/horovod:0.13.11-tf1.10.0-torch0.4.0-py3.5 .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/horovod:0.13.11-tf1.10.0-torch0.4.0-py3.5