FROM python:3.7

ARG PIP_NO_CACHE_DIR=1

# Download latest listing of available packages:
RUN apt-get -y update
# Upgrade already installed packages:
RUN apt-get -y upgrade
# Install a new package:
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx git
RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3 numpy Pillow opencv-python boto3 tqdm cython

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
