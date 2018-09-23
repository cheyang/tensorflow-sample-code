DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu gpu

cpu:
	$(DOCKER) build --no-cache -f Dockerfile.cpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:cpu .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:cpu
gpu: 
	$(DOCKER) build --no-cache -f Dockerfile.gpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:gpu .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:gpu
