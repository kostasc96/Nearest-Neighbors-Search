import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSH
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

def preprocess(text):
    ps = PorterStemmer() 
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    stemmed_tokens = []
    for t in tokens:
        stemmed_tokens.append(ps.stem(t))
    return stemmed_tokens

def get_forest(data, perms):
    start_time = time.time()
    minhash = []
    for text in data:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
    forest = MinHashLSH(threshold=0.8, num_perm=perms)
    for i,m in enumerate(minhash):
        forest.insert(i, m)
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    return forest

def predict(text, database, perms, forest):
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
    idx_array = np.array(forest.query(m))
    return len(idx_array)


def main():
    train = pd.read_csv('datasets/q2a/corpusTrain.csv')
    train = train[0:50000]
    test = pd.read_csv('datasets/q2a/corpusTest.csv')
    test = test[0:1000]
    train = train['Content']
    test = test['Content']
    permutations = 64
    forest = get_forest(train, permutations)
    results=[]
    start_time = time.time()
    for text in test:
        result = predict(text, train, permutations, forest)
        results.append(result)
    duplicates=0
    for x in results:
        duplicates = duplicates + x
    print ('number of duplicates is: {}'.format(duplicates))
    print('It took %s seconds to query forest.' %(time.time()-start_time))

if __name__== "__main__":
  main()
