WORKSPACE:=/gulf-slr

build-image:
	docker build . -t gulf-slr --target base

bash: 
	docker run -it --rm \
	--gpus all \
	-v ${CURDIR}:${WORKSPACE} \
	gulf-slr \
	bash