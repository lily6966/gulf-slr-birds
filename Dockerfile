FROM tensorflow/tensorflow:latest-gpu-jupyter AS builder

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

FROM tensorflow/tensorflow:latest-gpu-jupyter AS base

# copy over all python files from builder stage
COPY --from=builder /usr/local /usr/local

WORKDIR /gulf-slr 