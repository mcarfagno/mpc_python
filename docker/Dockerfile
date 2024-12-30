FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# install requirements
RUN pip install --no-cache-dir --requirement requirements.txt


# Copy the python project
WORKDIR ~/mpc_ws
COPY mpc_pybullet_demo ./

# Execute the command passed into this entrypoint
CMD ["exec", "$@"]
