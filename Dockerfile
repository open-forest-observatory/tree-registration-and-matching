# Python base image
from python:3.12-slim

# Install curl
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app

# Install the module dependencies with poetry without creating a virtual environment
RUN /root/.local/bin/poetry config virtualenvs.create false && /root/.local/bin/poetry install