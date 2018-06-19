DOCKER ?= docker

.NOTPARALLEL:
.PHONY: all

all: 
	$(DOCKER) build --no-cache -f Dockerfile -t registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-client-demo .
    $(DOCKER) push registry.cn-hangzhou.aliyuncs.com/tensorflow-samples/tf-mnist-client-demo