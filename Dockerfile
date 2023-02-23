FROM continuumio/miniconda3

WORKDIR /app
RUN conda create -y -n myenv
COPY . .
RUN pip install -r requirements.txt

# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# CMD gunicorn -b 0.0.0.0:80 app.app:server
EXPOSE 8000
ENTRYPOINT ["python", "/app/dashboard.py"]