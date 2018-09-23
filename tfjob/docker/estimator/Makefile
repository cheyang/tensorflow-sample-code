DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu gpu

cpu:
	$(DOCKER) build --no-cache -f Dockerfile.cpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/mnist-estimator:cpu .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/mnist-estimator:cpu
gpu: 
	$(DOCKER) build --no-cache -f Dockerfile.gpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/mnist-estimator:gpu .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/mnist-estimator:gpu
