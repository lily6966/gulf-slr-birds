WORKSPACE:=/gulf-slr

repo-init:
	python -m pip install pre-commit=3.4.0 && \
	pre-commit install

build-image:
	docker build . -t gulf-slr --target base

bash:
	docker run -it --rm \
	--gpus all \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-v ${CURDIR}:${WORKSPACE} \
	gulf-slr \
	bash
