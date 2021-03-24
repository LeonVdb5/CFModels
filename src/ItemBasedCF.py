import Algorithmia
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import pickle
import pandas as pd

client = Algorithmia.client()

def load_model():
    modelFile = client.file("data://lvandenb/CFModels/trained_knn_model.pkl").getFile().name
    with open(modelFile, 'rb') as f:
        model = pickle.load(f)
        return model

def load_train_data():
    dataFile = client.file("data://lvandenb/CFModels/training_data.npz").getFile().name
    training_data = sparse.load_npz(dataFile)
    return training_data
    
def load_movies():
    moviesFile = client.file("data://lvandenb/CFModels/movie_ids.npy").getFile().name
    movie_ids = np.load(moviesFile, allow_pickle=True)
    return movie_ids

knn = load_model()
training_data = load_train_data()
movie_ids = load_movies()


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    
    movie_idx = np.where(movie_ids == input)[0][0]

    # return "hello {}".format(input)
    dists, idxs = knn.kneighbors(training_data[movie_idx], n_neighbors=21)
    
    # Append [movieid, distance] for each movie to the similar_movies list
    similar_movies = []
    for dist, idx in zip(dists.squeeze(), idxs.squeeze()):
        similar_movies.append([movie_ids[idx], dist])
    
    # Sort similar_movies by similarity distance (greatest to lowest) and return
    similar_movies = sorted(similar_movies, key=lambda x: x[1])[::-1]

    return similar_movies[0:20]
