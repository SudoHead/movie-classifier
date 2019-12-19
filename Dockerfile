FROM jupyter/scipy-notebook

RUN conda install tensorflow-gpu cudatoolkit=9.0 nltk

COPY . /home/jovyan/work

CMD bash