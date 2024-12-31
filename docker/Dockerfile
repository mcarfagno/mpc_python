FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# these libraries are required for GLX support
RUN apt-get update && apt-get install -y \
    libxt6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libqt5widgets5 \
    xvfb \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-packages

WORKDIR /mpc_ws

# install requirements
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

# Copy the python project
COPY mpc_pybullet_demo ./

# Execute the command passed into this entrypoint
CMD ["exec", "$@"]
