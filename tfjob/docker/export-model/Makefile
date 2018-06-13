DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu

cpu:
	$(DOCKER) build -f Dockerfile.cpu -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/export-model .
        $(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/export-model
