FROM jupyter/scipy-notebook

WORKDIR /home/jovyan/work

COPY ./requirements.txt .

# Install the required libraries and copy files
RUN while IFS= read -r requirement; do conda install $requirement; done < requirements.txt
# RUN conda install tensorflow-gpu \
#     cudatoolkit=9.0 \
#     nltk

# RUN conda install inflect \
#     gensim

COPY . .

# Run a bash terminal when the containers is run
CMD bash