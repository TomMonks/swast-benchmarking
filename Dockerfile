# Use publicly available Linux-Miniconda image
# start with miniconda image
FROM continuumio/miniconda3:latest

# Create and set the working directory to the container make copy simpler
RUN mkdir /home/code
WORKDIR /home/code

# Copy all files across to container
COPY . /home/code

# Copy jupyter config file
COPY ./docker/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Install anaconda, conda-forge and pip dependencies
RUN conda env create -f binder/environment.yml && conda clean -afy --yes

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ambo_benchmark", "/bin/bash", "-c"]

# install svd and Rssa via cran.
RUN conda run -n ambo_benchmark Rscript docker/install_rssa.R

# Declare port used by jupyter-lab
EXPOSE 80
