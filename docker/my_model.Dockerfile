ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace/bench

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/bench

RUN python -m pip install --upgrade pip \
 && python -m pip install -e .

# Your model deps placeholder:
# RUN python -m pip install -r requirements_my_model.txt

CMD ["bash"]

