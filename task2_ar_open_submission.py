
# coding: utf-8
# Ayah Zirikly --varDial shared task (ar) [OPEN]
# Run1 accuracy results:
# C 0.4435
# includes system in ensemble (Run3)

# In[9]:

# Test Submission OPEN chngrams123456_msa
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

def loadDic(dic):
    return set(line.strip() for line in open(dic))

def inDict(dicFile, inputFile):
    dic = loadDic(dicFile)
    firstLine = True
    dicFeat = []
    words = []
    with open(inputFile, 'r') as f:
        for line in f:
            if firstLine:
                firstLine = False
                continue
            count = 0
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in dic:
                    count += 1
            dicFeat.append(count)
    return pd.DataFrame(dicFeat)

def ngrams_LR(goldTest, dictFile_MSA):
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
    train_df['inMSA'] = inDict(dictFile_MSA, train_file)
    
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    print test_df.shape
    test_df['inMSA'] = inDict(dictFile_MSA, test_file)
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(1,6), 
                                lowercase = False)
    
    # Save vectorizers 
    if goldTest:
        #pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'rb'))
    
    else:
        pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'wb'))
        # load vectorizer
        vectorize2 = pickle.load(open('../data/DSL2016-test/ar/models/chngrams123456.vectorizer', 'rb'))
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    X_msa = train_df[['inMSA']]
    
    X = hstack([X_chngrams, 
                X_msa, 
               ])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams123456_msa.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams123456_msa.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams123456_msa.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams123456_msa.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    X_msa_test = test_df[['inMSA']]
    
    X_test = hstack([X_chngrams_test, 
                     X_msa_test, 
                    ])
    
    predicted_y = clf2.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams123456_msa.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams123456_msa.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path_dic = '../data/DSL-training/ar/'
    dicFile_MSA = path_dic + 'resources/MSA.833M.cbow.vec.vocab'
    ngrams_LR(False, dicFile_MSA)


# In[11]:

# Test Submission CLOSE chngrams123456_ngram123_msa_egy_egyT_norT_lavT_glfT
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

def loadDic(dic):
    return set(line.strip() for line in open(dic))

def inDict(dicFile, inputFile):
    dic = loadDic(dicFile)
    firstLine = True
    dicFeat = []
    words = []
    with open(inputFile, 'r') as f:
        for line in f:
            if firstLine:
                firstLine = False
                continue
            count = 0
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in dic:
                    count += 1
            dicFeat.append(count)
    return pd.DataFrame(dicFeat)

def ngrams_LR(goldTest, dictFile_MSA, dictFile_EGY, 
              dicFile_EGY_twitter, dicFile_GLF_twitter, dicFile_NOR_twitter, dicFile_LAV_twitter):
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
    train_df['inMSA'] = inDict(dictFile_MSA, train_file)
    train_df['inEGY'] = inDict(dictFile_EGY, train_file)
    train_df['EGY_twitter'] = inDict(dicFile_EGY_twitter, train_file)
    train_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, train_file)
    train_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, train_file)
    train_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, train_file)
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    
    print test_df.shape
    test_df['inMSA'] = inDict(dictFile_MSA, test_file)
    test_df['inEGY'] = inDict(dictFile_EGY, test_file)
    test_df['EGY_twitter'] = inDict(dicFile_EGY_twitter, test_file)
    test_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, test_file)
    test_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, test_file)
    test_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, test_file)
    
    
    vectorize =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(1,6), 
                                lowercase = False)
    vectorize_word =  CountVectorizer(tokenizer=None, analyzer = 'word', ngram_range=(1,3), 
                                lowercase = False)
    
    # Save vectorizers 
    if goldTest:
        #pickle.dump(vectorize, open('../data/DSL2016-test/ar/models/dev/chngrams123456.vectorizer', 'wb'))
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
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word2.fit_transform(train_df['sentence'])
    X_msa = train_df[['inMSA']]
    X_egy = train_df[['inEGY']]
    X_egy_twitter = train_df[['EGY_twitter']]
    X_lav_twitter = train_df[['LAV_twitter']]
    X_nor_twitter = train_df[['NOR_twitter']]
    X_glf_twitter = train_df[['GLF_twitter']]
    
    X = hstack([X_chngrams, X_ngrams_w, 
                X_msa, 
                X_egy, 
                X_egy_twitter, 
                X_lav_twitter, 
                X_nor_twitter, 
                X_glf_twitter
               ])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word2.transform(test_df['sentence'])
    X_msa_test = test_df[['inMSA']]
    X_egy_test = test_df[['inEGY']]
    X_egy_twitter_test = test_df[['EGY_twitter']]
    X_lav_twitter_test = test_df[['LAV_twitter']]
    X_nor_twitter_test = test_df[['NOR_twitter']]
    X_glf_twitter_test = test_df[['GLF_twitter']]
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test, 
                     X_msa_test, 
                     X_egy_test, 
                     X_egy_twitter_test, 
                     X_lav_twitter_test, 
                     X_nor_twitter_test, 
                     X_glf_twitter_test
                    ])
    
    predicted_y = clf2.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams123456_ngrams123_msa_egy_egyT_norT_lavT_glfT.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path_dic = '../data/DSL-training/ar/'
    dicFile_MSA = path_dic + 'resources/MSA.833M.cbow.vec.vocab'
    dicFile_EGY = path_dic + 'resources/tharwa.egyWords'
    dicFile_EGY_twitter = path_dic + 'resources/egy_twitter_lexicon.txt'
    dicFile_GLF_twitter = path_dic + 'resources/glf_twitter_lexicon.txt'
    dicFile_NOR_twitter = path_dic + 'resources/nor_twitter_lexicon.txt'
    dicFile_LAV_twitter = path_dic + 'resources/lav_twitter_lexicon.txt'
    
    ngrams_LR(False, 
              dicFile_MSA, 
              dicFile_EGY, 
              dicFile_EGY_twitter, 
              dicFile_GLF_twitter, 
              dicFile_NOR_twitter, 
              dicFile_LAV_twitter)


# In[23]:

# Test Submission CLOSE chngrams123456_ngram123_msa_egy_norT_lavT_glfT
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

def loadDic(dic):
    return set(line.strip() for line in open(dic))

def inDict(dicFile, inputFile):
    dic = loadDic(dicFile)
    firstLine = True
    dicFeat = []
    words = []
    with open(inputFile, 'r') as f:
        for line in f:
            if firstLine:
                firstLine = False
                continue
            count = 0
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in dic:
                    count += 1
            dicFeat.append(count)
    return pd.DataFrame(dicFeat)

def ngrams_LR(goldTest, dictFile_MSA, dictFile_EGY, 
              dicFile_GLF_twitter, dicFile_NOR_twitter, dicFile_LAV_twitter):
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
    train_df['inMSA'] = inDict(dictFile_MSA, train_file)
    train_df['inEGY'] = inDict(dictFile_EGY, train_file)
    train_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, train_file)
    train_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, train_file)
    train_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, train_file)
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    
    print test_df.shape
    test_df['inMSA'] = inDict(dictFile_MSA, test_file)
    test_df['inEGY'] = inDict(dictFile_EGY, test_file)
    test_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, test_file)
    test_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, test_file)
    test_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, test_file)
    
    
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
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word2.fit_transform(train_df['sentence'])
    X_msa = train_df[['inMSA']]
    X_egy = train_df[['inEGY']]
    X_lav_twitter = train_df[['LAV_twitter']]
    X_nor_twitter = train_df[['NOR_twitter']]
    X_glf_twitter = train_df[['GLF_twitter']]
    
    X = hstack([X_chngrams, X_ngrams_w, 
                X_msa, 
                X_egy, 
                X_lav_twitter, 
                X_nor_twitter, 
                X_glf_twitter
               ])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word2.transform(test_df['sentence'])
    X_msa_test = test_df[['inMSA']]
    X_egy_test = test_df[['inEGY']]
    X_lav_twitter_test = test_df[['LAV_twitter']]
    X_nor_twitter_test = test_df[['NOR_twitter']]
    X_glf_twitter_test = test_df[['GLF_twitter']]
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test, 
                     X_msa_test, 
                     X_egy_test, 
                     X_lav_twitter_test, 
                     X_nor_twitter_test, 
                     X_glf_twitter_test
                    ])
    
    predicted_y = clf2.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path_dic = '../data/DSL-training/ar/'
    dicFile_MSA = path_dic + 'resources/MSA.833M.cbow.vec.vocab'
    dicFile_EGY = path_dic + 'resources/tharwa.egyWords'
    dicFile_GLF_twitter = path_dic + 'resources/glf_twitter_lexicon.txt'
    dicFile_NOR_twitter = path_dic + 'resources/nor_twitter_lexicon.txt'
    dicFile_LAV_twitter = path_dic + 'resources/lav_twitter_lexicon.txt'
    
    ngrams_LR(True, 
              dicFile_MSA, 
              dicFile_EGY, 
              dicFile_GLF_twitter, 
              dicFile_NOR_twitter, 
              dicFile_LAV_twitter)


# In[33]:

# Test Submission OPEN chngrams23456_ngram123_msa_egy_norT_lavT_glfT
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

def loadDic(dic):
    return set(line.strip() for line in open(dic))

def inDict(dicFile, inputFile):
    dic = loadDic(dicFile)
    firstLine = True
    dicFeat = []
    words = []
    with open(inputFile, 'r') as f:
        for line in f:
            if firstLine:
                firstLine = False
                continue
            count = 0
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in dic:
                    count += 1
            #dicFeat.append(count)
            dicFeat.append(round(count *1.0 / len(words), 1))
    return pd.DataFrame(dicFeat)

def ngrams_LR(goldTest, dictFile_MSA, dictFile_EGY, 
              dicFile_GLF_twitter, dicFile_NOR_twitter, dicFile_LAV_twitter):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample2_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample2_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    train_df['inMSA'] = inDict(dictFile_MSA, train_file)
    train_df['inEGY'] = inDict(dictFile_EGY, train_file)
    train_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, train_file)
    train_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, train_file)
    train_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, train_file)
    
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    test_df['inMSA'] = inDict(dictFile_MSA, test_file)
    test_df['inEGY'] = inDict(dictFile_EGY, test_file)
    test_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, test_file)
    test_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, test_file)
    test_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, test_file)
    
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
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word2.fit_transform(train_df['sentence'])
    X_msa = train_df[['inMSA']]
    X_egy = train_df[['inEGY']]
    X_lav_twitter = train_df[['LAV_twitter']]
    X_nor_twitter = train_df[['NOR_twitter']]
    X_glf_twitter = train_df[['GLF_twitter']]
    
    X = hstack([X_chngrams, X_ngrams_w, X_msa, X_egy, 
                X_lav_twitter, 
                X_nor_twitter, 
                X_glf_twitter
               ])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word2.transform(test_df['sentence'])
    X_msa_test = test_df[['inMSA']]
    X_egy_test = test_df[['inEGY']]
    X_lav_twitter_test = test_df[['LAV_twitter']]
    X_nor_twitter_test = test_df[['NOR_twitter']]
    X_glf_twitter_test = test_df[['GLF_twitter']]
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test, X_msa_test, X_egy_test, 
                     X_lav_twitter_test, 
                     X_nor_twitter_test, 
                     X_glf_twitter_test
                    ])
    
    predicted_y = clf.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams23456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path_dic = '../data/DSL-training/ar/'
    dicFile_MSA = path_dic + 'resources/MSA.833M.cbow.vec.vocab'
    dicFile_EGY = path_dic + 'resources/tharwa.egyWords'
    dicFile_GLF_twitter = path_dic + 'resources/glf_twitter_lexicon.txt'
    dicFile_NOR_twitter = path_dic + 'resources/nor_twitter_lexicon.txt'
    dicFile_LAV_twitter = path_dic + 'resources/lav_twitter_lexicon.txt'
    
    ngrams_LR(False, 
              dicFile_MSA, 
              dicFile_EGY, 
              dicFile_GLF_twitter, 
              dicFile_NOR_twitter, 
              dicFile_LAV_twitter)


# In[34]:

# Test Submission OPEN chngrams123456_ngram123_msa_egy_norT_lavT_glfT
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

def loadDic(dic):
    return set(line.strip() for line in open(dic))

def inDict(dicFile, inputFile):
    dic = loadDic(dicFile)
    firstLine = True
    dicFeat = []
    words = []
    with open(inputFile, 'r') as f:
        for line in f:
            if firstLine:
                firstLine = False
                continue
            count = 0
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in dic:
                    count += 1
            #dicFeat.append(count)
            dicFeat.append(round(count *1.0 / len(words), 1))
    return pd.DataFrame(dicFeat)

def ngrams_LR(goldTest, dictFile_MSA, dictFile_EGY, 
              dicFile_GLF_twitter, dicFile_NOR_twitter, dicFile_LAV_twitter):
    # Load the data
    if goldTest:
        train_file = "../data/DSL-training/ar/task2-train-sample2_header.txt"
        test_file  = "../data/DSL-training/ar/task2-test-sample2_header.txt"
    
    else:
        train_file = "../data/DSL-training/ar/task2-train_header.txt"
        test_file  = "../data/DSL2016-test/ar/C_header.txt"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, skip_blank_lines=True)
    train_df = train_df[pd.notnull(train_df['sentence'])]
    train_df['inMSA'] = inDict(dictFile_MSA, train_file)
    train_df['inEGY'] = inDict(dictFile_EGY, train_file)
    train_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, train_file)
    train_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, train_file)
    train_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, train_file)
    
    test_df = pd.read_csv(test_file, sep="\t", header = 0)
    test_df['inMSA'] = inDict(dictFile_MSA, test_file)
    test_df['inEGY'] = inDict(dictFile_EGY, test_file)
    test_df['LAV_twitter'] = inDict(dicFile_LAV_twitter, test_file)
    test_df['NOR_twitter'] = inDict(dicFile_NOR_twitter, test_file)
    test_df['GLF_twitter'] = inDict(dicFile_GLF_twitter, test_file)
    
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
    
    X_chngrams = vectorize2.fit_transform(train_df['sentence'])
    X_ngrams_w = vectorize_word2.fit_transform(train_df['sentence'])
    X_msa = train_df[['inMSA']]
    X_egy = train_df[['inEGY']]
    X_lav_twitter = train_df[['LAV_twitter']]
    X_nor_twitter = train_df[['NOR_twitter']]
    X_glf_twitter = train_df[['GLF_twitter']]
    
    X = hstack([X_chngrams, X_ngrams_w, X_msa, X_egy, 
                X_lav_twitter, 
                X_nor_twitter, 
                X_glf_twitter
               ])
    
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, train_df['DA'])
    
    if goldTest:
         # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/dev/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))
    
    else:
        # Save model
        pickle.dump(clf, 
                    open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'wb'))
        #load model
        clf2 = pickle.load(
            open('../data/DSL2016-test/ar/models/chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.clf', 'rb'))

    X_chngrams_test = vectorize2.transform(test_df['sentence'])
    X_ngrams_w_test = vectorize_word2.transform(test_df['sentence'])
    X_msa_test = test_df[['inMSA']]
    X_egy_test = test_df[['inEGY']]
    X_lav_twitter_test = test_df[['LAV_twitter']]
    X_nor_twitter_test = test_df[['NOR_twitter']]
    X_glf_twitter_test = test_df[['GLF_twitter']]
    
    X_test = hstack([X_chngrams_test, X_ngrams_w_test, X_msa_test, X_egy_test, 
                     X_lav_twitter_test, 
                     X_nor_twitter_test, 
                     X_glf_twitter_test
                    ])
    
    predicted_y = clf.predict(X_test)
    
    path = "../data/DSL2016-test/ar/predictions/"
    
    if (goldTest):
        y_test = test_df['DA']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        output_file = open(path + 'dev_open_chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        output_file.write('sentence\tpredicted\tgold\n')
        for pid, py, goldY in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(str(pid) +'\t' + py + '\t' + goldY + '\n')
        output_file.close()
    
    else:
        output_file = open(path + 'test_open_chngrams123456_ngrams123_msa_egy_norT_lavT_glfT.tsv','w')
        for pid, py in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(str(pid) +'\t' + py + '\n')
        output_file.close()

if __name__ == "__main__":
    path_dic = '../data/DSL-training/ar/'
    dicFile_MSA = path_dic + 'resources/MSA.833M.cbow.vec.vocab'
    dicFile_EGY = path_dic + 'resources/tharwa.egyWords'
    dicFile_GLF_twitter = path_dic + 'resources/glf_twitter_lexicon.txt'
    dicFile_NOR_twitter = path_dic + 'resources/nor_twitter_lexicon.txt'
    dicFile_LAV_twitter = path_dic + 'resources/lav_twitter_lexicon.txt'
    
    ngrams_LR(False, 
              dicFile_MSA, 
              dicFile_EGY, 
              dicFile_GLF_twitter, 
              dicFile_NOR_twitter, 
              dicFile_LAV_twitter)


# In[ ]:



