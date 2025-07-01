# Dockerfile for building a conda environment with CLIP dependencies
FROM continuumio/miniconda3:latest

# set environment variables
WORKDIR /envbuild

# copy environment file
COPY env.yml .

# install conda environment
RUN conda update -n base -c defaults conda && \
    conda env create -f env.yml && \
    conda clean -afy

# activate the conda environment
SHELL ["conda", "run", "--no-capture-output", "-n", "clip", "/bin/bash", "-c"]

# start interactive shell   
CMD [ "bash" ]

