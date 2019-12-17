FROM jupyter/scipy-notebook

RUN conda install tensorflow-gpu cudatoolkit=9.0

COPY . /home/jovyan/work

CMD bash