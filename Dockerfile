FROM jupyter/scipy-notebook

RUN pip install joblib

COPY Data/name_gender.csv Data/name_gender.csv
COPY Data/unseen_names.csv Data/unseen_names.csv

COPY train.py ./train.py
COPY infer_on_unseen.py ./infer_on_unseen.py

RUN python3 train.py