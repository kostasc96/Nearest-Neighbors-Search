import numpy as np
import pandas as pd
from numpy import dot
from numpy import random
import librosa
import os
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD

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
        
class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))

def cosine_sim(vec1, vec2):
        return dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def main():
    train = pd.read_csv('datasets/q2a/corpusTrain.csv')
    train = train[0:4000]
    ids_train = train['Id']
    contents_train = train['Content']

    test = pd.read_csv('datasets/q2a/corpusTest.csv')
    test = test[0:1200]
    ids_test = test['Id']
    contents_test = test['Content']

    components = 500

    vectorizer = TfidfVectorizer(stop_words = 'english', use_idf=True)
    lsa=TruncatedSVD(n_components = components)
    svd_transformer = make_pipeline(vectorizer,lsa)
    vector = svd_transformer.fit_transform(contents_train)
    array_vector_train= vector

    vector = svd_transformer.fit_transform(contents_test)
    array_vector_test= vector

    hash_table = HashTable(hash_size=8, inp_dimensions=components)

    counter = 0
    for id in ids_train:
        hash_table.__setitem__(array_vector_train[counter], id)
        counter = counter + 1

    #print (hash_table.hash_table.keys())

    match_for_every_test = []
    for i in range(0,len(array_vector_test)):
        key = hash_table.generate_hash(array_vector_test[i])
        keys = list(hash_table.hash_table[key])
        match = 0
        for key in keys:
            distance = cosine_sim(array_vector_train[key], array_vector_test[i])
            if distance > 0.8:
                match = match + 1
        match_for_every_test.append(match)
    
    for item in match_for_every_test:
        if item != 0:
            print ('hello')
            print(item)

if __name__== "__main__":
  main()
