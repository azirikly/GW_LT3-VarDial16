
# coding: utf-8

# In[1]:
# Ayah Zirikly --varDial shared task (ar) [CLOSE]
# Run1 accuracy results:
# C 0.4442
# includes system in ensemble (Run3)

# Test Submission CLOSE chngrams1-6
import numpy as np
import re, math
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from StringIO import StringIO
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack 
from sklearn import preprocessing
import pickle 

def ngrams_LR(goldTest):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample2_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample2_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    print train_df.shape
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    print test_df.shape
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(1,6), 
                                lowercase = False)
    if (goldTest):
        # Save vectorizer
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'rb'))
    else:
        # Save vectorizer
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'rb'))
        
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_chngrams, train_df['DA'])
    
    if (goldTest):
        # Save model
        pickle.dump(clf, open('../data/DSL2016-test/ar/models/dev/chngrams123456.clf', 'wb'))
        #load model
        clf2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/chngrams123456.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, open('../data/DSL2016-test/ar/models/chngrams123456.clf', 'wb'))
        #load model
        clf2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams123456.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    predicted_y = clf2.predict(X_chngrams_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_chngrams123456.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_close_chngrams123456.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path = '../data/DSL-training/ar/'
    ngrams_LR(True)


# In[ ]:

# Test Submission CLOSE chngrams2-6
import numpy as np
import re, math
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from StringIO import StringIO
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack 
from sklearn import preprocessing
import pickle 

def ngrams_LR(goldTest):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample2_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample2_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    print train_df.shape
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(2,6), 
                                lowercase = False)
    # Save vectorizer
    pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/chngrams23456.vectorizer', 'wb'))
    
    # load vectorizer
    vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams23456.vectorizer', 'rb'))
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_chngrams, train_df['DA'])
    
    # Save model
    pickle.dump(clf, open('../data/DSL2016-test/ar/models/chngrams23456.clf', 'wb'))
    
    #load model
    clf2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams23456.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    predicted_y = clf2.predict(X_chngrams_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_chngrams23456.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_close_chngrams23456.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path = '../data/DSL-training/ar/'
    ngrams_LR(False)


# In[11]:

# Test Submission CLOSE chngrams123456_ngrams123
import numpy as np
import numpy as np
import re, math
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from StringIO import StringIO
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack 
from sklearn import preprocessing
import pickle 

def ngrams_LR(goldTest):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample2_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample2_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(1,6), 
                                lowercase = False)
    vectorize_word =  CountVectorizer(tokenizer=None, analyzer = 'word', ngram_range=(1,3), 
                                lowercase = False)
    
    # Save vectorizers 
    if goldTest:
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'wb'))
        pickle.dump(vectorize_word, open('../data/DSL2016-test/ar/models/dev/ngrams123.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'rb'))
        vectorize_word2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/ngrams123.vectorizer', 'rb'))
    
    else:
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'wb'))
        pickle.dump(vectorize_word, open('../data/DSL2016-test/ar/models/ngrams123.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'rb'))
        vectorize_word2 = pickle.load(open('../data/DSL2016-test/ar/models/ngrams123.vectorizer', 'rb'))
    
    X_chngrams = vectorize.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word.fit_transform(train_df['sentence'])
    
    X = hstack([X_chngrams, X_ngrams_w])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123.clf', 'rb'))

    X_chngrams_test = vectorize.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word.transform(test_df['sentence'])
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test])
    
    predicted_y = clf.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams123456_ngrams123.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams123456_ngrams123.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    ngrams_LR(True)


# In[17]:

# Test Submission CLOSE chngrams23456_ngrams123
import numpy as np
import numpy as np
import re, math
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from StringIO import StringIO
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack 
from sklearn import preprocessing
import pickle 

def ngrams_LR(goldTest):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(2,6), 
                                lowercase = False)
    vectorize_word =  CountVectorizer(tokenizer=None, analyzer = 'word', ngram_range=(1,3), 
                                lowercase = False)
    
    # Save vectorizers 
    if goldTest:
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/dev/chngrams23456.vectorizer', 'wb'))
        pickle.dump(vectorize_word, open('../data/DSL2016-test/ar/models/dev/ngrams123.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/chngrams23456.vectorizer', 'rb'))
        vectorize_word2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/ngrams123.vectorizer', 'rb'))
    
    else:
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/chngrams23456.vectorizer', 'wb'))
        pickle.dump(vectorize_word, open('../data/DSL2016-test/ar/models/ngrams123.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams23456.vectorizer', 'rb'))
        vectorize_word2 = pickle.load(open('../data/DSL2016-test/ar/models/ngrams123.vectorizer', 'rb'))
    
    X_chngrams = vectorize.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word.fit_transform(train_df['sentence'])
    
    X = hstack([X_chngrams, X_ngrams_w])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams23456_ngrams123.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams23456_ngrams123.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams23456_ngrams123.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams23456_ngrams123.clf', 'rb'))

    X_chngrams_test = vectorize.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word.transform(test_df['sentence'])
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test])
    
    predicted_y = clf.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_close_chngrams23456_ngrams123.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_close_chngrams23456_ngrams123.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    ngrams_LR(False)


# In[ ]:



