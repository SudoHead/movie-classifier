FROM jupyter/scipy-notebook

WORKDIR /home/jovyan/work

COPY ./requirements.txt .
COPY ./install_nltk_corpus.py .

# Install the required libraries and copy files
# RUN while IFS= read -r requirement; do conda install $requirement; done < requirements.txt
RUN pip install -r requirements.txt
RUN python install_nltk_corpus.py

COPY . .

RUN python setup.py install

# Run a bash terminal when the containers is run
CMD bash