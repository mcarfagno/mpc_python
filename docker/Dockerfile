FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

WORKDIR /mpc_ws

# install requirements
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

# Copy the python project
COPY mpc_pybullet_demo ./

# Execute the command passed into this entrypoint
CMD ["exec", "$@"]
