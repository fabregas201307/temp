FROM python:3.7

ARG PIP_NO_CACHE_DIR=1

# Download latest listing of available packages:
RUN apt-get -y update
# Upgrade already installed packages:
RUN apt-get -y upgrade
# Install a new package:
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
RUN pip3 install numpy>=1.18 opencv-python>=4.2 boto3>=1.14 tifffile==2020.9.3 imagecodecs==2020.5.30

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
