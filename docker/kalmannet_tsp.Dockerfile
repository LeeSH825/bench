ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace/bench

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy bench (in practice, you may mount instead; this is for image completeness)
COPY . /workspace/bench

# Install bench common deps (torch is assumed to exist in BASE_IMAGE)
RUN python -m pip install --upgrade pip \
 && python -m pip install -e . \
 && if [ -f requirements.lock ]; then echo "Found requirements.lock"; fi

# Model-specific deps placeholder (to be filled after third_party attached)
# RUN python -m pip install -r third_party/KalmanNet_TSP/requirements.txt

CMD ["bash"]
