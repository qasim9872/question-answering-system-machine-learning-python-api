FROM python:2

RUN apt-get update && apt-get install -y git

# Create and move inside directory
COPY . /code
WORKDIR /code

# Install packages
RUN pip install -r requirements.txt

# Install packages for sub module
RUN cd nmt-keras && pip install -r requirements.txt && cd ..

# checkout the branch required to run the web server in the nmt package
RUN cd nmt-keras/src/keras-wrapper && git checkout Interactive_NMT && cd ../../..

ENTRYPOINT [ "python", "nmt-keras/demo-web/sample_server.py" ]
CMD ["-c", "music-nmt-model/model_minor/config.pkl", "-ds", "music-nmt-model/datasets/Dataset_music-dataset_ensparql.pkl", "-m", "music-nmt-model/model_minor/epoch_500"]




