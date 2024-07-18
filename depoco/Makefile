IMAGE_NAME=depoco
TAG=latest
DATASETS=/home/rama/data/a9_dataset_r02_s02

build:
	@echo Building docker container $(IMAGE_NAME)
	nvidia-docker build -t $(IMAGE_NAME):$(TAG) .

test:
	@echo NVIDIA and CUDA setup
	@nvidia-docker run --rm $(IMAGE_NAME):$(TAG) nvidia-smi
	@echo PytTorch CUDA setup installed?
	@nvidia-docker run --rm $(IMAGE_NAME):$(TAG) python3 -c "import torch; print(torch.cuda.is_available())"

run:
	docker run --rm --gpus all -it -v $(DATASETS):/data -v /home/rama/depoco/package/compression:/depoco $(IMAGE_NAME)

clean:
	@echo Removing docker image...
	-docker image rm --force $(IMAGE_NAME):$(TAG)
