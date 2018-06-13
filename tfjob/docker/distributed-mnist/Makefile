DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu gpu

cpu:
	$(DOCKER) build -f Dockerfile.cpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:cpu-export .
    $(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:cpu-export
gpu: 
	$(DOCKER) build -f Dockerfile.gpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:gpu-export .
	$(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-distributed:gpu-export