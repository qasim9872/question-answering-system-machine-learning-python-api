FROM python:2.7.15

RUN apt-get update && apt-get install -y git

# Create and move inside directory
COPY . /code
WORKDIR /code

# Install packages
RUN pip install -r pip-requirements.txt

# checkout the branch required to run the web server in the nmt package
RUN cd ./src/keras-wrapper && git checkout Interactive_NMT && cd ../..
# Checking out to a specific version since the above checkout pulls the latest code from the project repo
RUN cd ./src/keras-wrapper && git checkout b8667e5abbb8ad4f471913e24a89bf343078576e  && cd ../..

ENTRYPOINT [ "python", "nmt-keras/demo-web/sample_server.py" ]
CMD ["-c", "music-nmt-model/model_minor/config.pkl", "-ds", "music-nmt-model/datasets/Dataset_music-dataset_ensparql.pkl", "-m", "music-nmt-model/model_minor/epoch_500"]
