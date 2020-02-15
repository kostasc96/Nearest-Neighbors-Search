import numpy as np
import pandas as pd
from numpy import dot
from numpy import random
import os
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
            .get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


def main():
    train = pd.read_csv('datasets/q2a/corpusTrain.csv')
    train = train[0:4000]
    ids_train = train['Id']
    contents_train = train['Content']

    test = pd.read_csv('datasets/q2a/corpusTest.csv')
    test = test[0:500]
    contents_test = test['Content']

    vectorizer = TfidfVectorizer(stop_words='english')
    vector = vectorizer.fit_transform(contents_train)
    array_vector_train = vector.toarray()

    components = array_vector_train.shape[1]

    vector = vectorizer.transform(contents_test)
    array_vector_test = vector.toarray()


    hash_table = HashTable(hash_size=5, inp_dimensions=components)
    t0 = time.time()

    counter = 0
    for id in ids_train:
        hash_table.__setitem__(array_vector_train[counter], id)
        counter = counter + 1
    t1 = time.time()
    print('Time to build LSH forest for train is: {} '.format(t1-t0))

    t2 = time.time()

    match_for_every_test = []
    for i in range(0, len(array_vector_test)):
        key = hash_table.generate_hash(array_vector_test[i])
        keys = list(hash_table.hash_table[key])
        match = 0
        for key in keys:
            distance = cosine_similarity([array_vector_train[key]], [array_vector_test[i]])
            if distance > 0.8:
                match = match + 1
        match_for_every_test.append(match)

    t3 = time.time()
    print('Time to query test set is: {} '.format(t3-t2))

    count = 0
    for item in match_for_every_test:
        count = count + item
    print('Similar items from test set to train set is : {}'.format(count))
    print('Total time is : {}'.format(t3-t0))


if __name__ == "__main__":
    main()

