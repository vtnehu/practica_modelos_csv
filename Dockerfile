FROM jupyter/scipy-notebook


# Install another dependency using pip
RUN pip install joblib

# Create folder to save models
RUN mkdir models

RUN mkdir datos

# Copy data into image
COPY ./datos/train.csv ./datos/train.csv

# Copy model into image
COPY ./train.py ./train.py






