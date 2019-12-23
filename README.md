# movie-classifier

## Introduction

Multi-label classification problem involving prediction of a movie's genres given a title and a text description.

#### Dataset

[MovieLens dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv).

#### Approach

The aim of this project is to build a ML system capable of assigning a set of label tags based on a name and a natural language text. Since a movie can have multiple genres ([Action, Comedy, Drama]), we cannot threat the problem as a multi-class classification problem. Instead, we can simplify it by generalising the problem to multiple binary classifications: commonly known as the one-vs-rest approach.

The scikit-learn library has been chosen for this project because it enables quick prototyping by offering a wide range of ML algorithms and a easy to use API. In particular, the implemented model (**OvRModel**) is a wrapper of the **sklearn.multiclass.OneVsRestClassifier** strategy. The advantage of this method is that it can use any type of classifier that inherits the sklearn's **BaseEstimator** class.

Furthermore, special attention has been given to the extendibility of the project: the use of an abstract class for the Model makes the integration of future ML algorithms a very straightforward task.

In order to transform the raw text data into useful features, we must apply a series of processing techniques which include:

- Convert to lowercase.
- Remove non-ASCII characters.
- Remove special characters like punctuation and extra spaces.
- Remove common stopwords.
- Convert numbers to text.
- Lemmatisation.

Then the TF-IDF algorithm is used to vectorize the transformed text, obtaining a feature vector that can be used to train the model.

The base estimator used in this project is the logistic regression, which achieves the following results using a threshold value of 0.3:

| Precision | Recall | F1-score |
|-----------|--------|----------|
| 0.557     | 0.626  | 0.589    |

## Installation

The ```requirements.txt``` file lists all the required libraries, to install them:

```
pip install -r requirements.txt
```

Run the ```setup.py``` to install the packages:

```
pip install .
```

### Docker

Alternatively, you can run a docker container. First, build the image from the Dockerfile:

```
docker image build -t pylearn .
```

Then run the container using:
```
docker run -it --name movie-classifier-app pylearn
```

Note: add ```-p 8888:8888``` and ```-v $pwd'':/home/jovyan/work``` if you want to run jupyter notebooks and mount the volume for persistency.

## Usage

The following commands assumes that the current directory is ```/src```.

### Inference

Run the script ```movie_classifier.py``` to make predictions:
```
python movie_classifier.py --title "The Shawshank Redemption" --description "In 1947 Portland, Maine, banker Andy Dufresne is convicted of murdering his wife and her lover and is sentenced to two consecutive life sentences at the Shawshank State Penitentiary. He is befriended by Ellis Red Redding, an inmate and prison contraband smuggler serving a life sentence."
```

### Data preprocessing

To clean the raw dataset (as csv), use the ```prepare_data.py``` script:
```
python prepare_data.py
```
Note: use -f "PATH" to specify the raw dataset, -s "PATH" to indicate where to save.

### Training

Although a trained model is included in this repository (```models/model.hal```), you can train a new one by running the ```training.py``` script:

```
python training.py
```

Note: use -f "PATH" to specify the cleaned dataset, -s "PATH" to indicate where to save the model, and --testsize # to set the proportion of the test set.

## Testing (unittest)

You can run the unittests from the ```/tests``` direcotory using:

```
python -m unittest test_*.py
```
