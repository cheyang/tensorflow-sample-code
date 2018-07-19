DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu gpu

cpu:
	$(DOCKER) build --no-cache -f Dockerfile.cpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed-v1alpha2:cpu .
    $(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed-v1alpha2:cpu
gpu: 
	$(DOCKER) build --no-cache -f Dockerfile.gpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed-v1alpha2:gpu .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed-v1alpha2:gpu
