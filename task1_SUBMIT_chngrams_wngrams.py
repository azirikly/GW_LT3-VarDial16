
# coding: utf-8

# Ayah Zirikly --varDial shared task [CLOSE]
# Run1 accuracy results:
# A 0.8859
# B1 0.92
# B2 0.878

# In[1]:

import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from StringIO import StringIO
from sklearn.svm import SVC
from scipy.sparse import hstack 
import csv, codecs, pickle

def ngrams_LR(goldDA):
    # Load the data
    train_file = "../data/DSL-training/task1/task1-trainDev-coarse_header.txt"
    test_file = "../data/DSL2016-test/A.txt"
    #test_file = "../data/DSL2016-test/B1.norm.filtered.txt"
    #test_file = "../data/DSL2016-test/B2.norm.filtered.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    test_df = pd.read_csv(test_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    print train_df.shape
    print test_df.shape
    vectorize_char =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(2,6), 
                                lowercase = True)
    vectorize_word =  CountVectorizer(tokenizer=None, analyzer = 'word', ngram_range=(1,3), 
                                lowercase = True)
    print 'start'
    train_chngrams_features = vectorize_char.fit_transform(train_df['sentence'])
    train_wngrams_features  = vectorize_word.fit_transform(train_df['sentence'])
    X = hstack([train_chngrams_features, train_wngrams_features])

    test_ngrams_features = vectorize_char.transform(test_df['sentence'])
    test_wngrams_features  = vectorize_word.transform(test_df['sentence'])
    X_test = hstack([test_ngrams_features, test_wngrams_features])
    
    print 'vectorize'
    #load model
    #clf = pickle.load(open('../data/DSL-training/task1/chngrams_wngrams/model/chngrams23456ngrams123.clf', 'rb'))
    clf = LogisticRegression(class_weight='balanced')
    #clf.fit(train_data_features.fillna(0), train_df['DA'])
    clf.fit(X, train_df['fine'])
    print 'fit'
    predicted_y = clf.predict(X_test)
    
    if goldDA:
        y_test = test_df['fine']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = codecs.open('../data/DSL-training/task1/dev_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
        output_file.write('sentence\tpredicted\tgold\n')
        for item in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(u"\t".join(item) + u"\n")
        output_file.close()
    else:
        output_file = codecs.open('../data/DSL-training/task1/chngrams_wngrams/A_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
        #output_file = codecs.open('../data/DSL-training/task1/chngrams_wngrams/B1_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
        #output_file = codecs.open('../data/DSL-training/task1/chngrams_wngrams/B2_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')

        for item in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(u"\t".join(item) + u"\n")
        output_file.close()
    
    test_file = "../data/DSL2016-test/B1.norm.filtered.txt"
    test_df = pd.read_csv(test_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    print 'start'
    test_ngrams_features = vectorize_char.transform(test_df['sentence'])
    test_wngrams_features  = vectorize_word.transform(test_df['sentence'])
    X_test = hstack([test_ngrams_features, test_wngrams_features])
    print 'vectorize'
    predicted_y = clf.predict(X_test)
    output_file = codecs.open('../data/DSL-training/task1/chngrams_wngrams/B1_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
    for item in zip(test_df['sentence'].tolist(), predicted_y):
        output_file.write(u"\t".join(item) + u"\n")
    output_file.close()
    
    test_file = "../data/DSL2016-test/B2.norm.filtered.txt"
    test_df = pd.read_csv(test_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    print 'start'
    test_ngrams_features = vectorize_char.transform(test_df['sentence'])
    test_wngrams_features  = vectorize_word.transform(test_df['sentence'])
    X_test = hstack([test_ngrams_features, test_wngrams_features])
    print 'vectorize'
    predicted_y = clf.predict(X_test)
    output_file = codecs.open('../data/DSL-training/task1/chngrams_wngrams/B2_out_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
    for item in zip(test_df['sentence'].tolist(), predicted_y):
        output_file.write(u"\t".join(item) + u"\n")
    output_file.close()
    
    #save model
    pickle.dump(clf, open('../data/DSL-training/task1/chngrams_wngrams/model/chngrams23456ngrams123.clf', 'wb'))
    
if __name__ == "__main__":
    ngrams_LR(False)


# In[ ]:



