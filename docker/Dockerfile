FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install --upgrade pip
RUN pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install python dependnecies
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

WORKDIR /bayesian_inference
