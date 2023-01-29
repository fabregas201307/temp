FROM davidwu20/xgboost:latest

RUN apt-get update && apt-get -y update

# RUN apt-get update

RUN pip3 -q install pip --upgrade
RUN mkdir src
WORKDIR src/
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter
RUN pip3 install "dask[complete]"
RUN apt-get -y install git
RUN git clone https://github.com/fabregas201307/alphalens.git
# RUN cd alphalens
RUN pip3 install -e ./alphalens/.
# WORKDIR /src/notebooks

# RUN python -m pip install "dask[complete]"    # Install everything

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
