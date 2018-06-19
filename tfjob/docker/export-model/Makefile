DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: cpu

cpu:
    $(DOCKER) build --no-cache -f Dockerfile -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/export-model .
    $(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/export-model