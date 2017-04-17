
# coding: utf-8
# Ayah Zirikly --varDial shared task [CLOSE]
# Run3 accuracy results:
# A 0.887
# B1 0.912
# B2 0.872

# In[9]:

# Create Coarse classifier to predict coarse label
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
    test_file = "../data/DSL2016-test/B2.norm.filtered.txt"
    
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

    test_chngrams_features = vectorize_char.transform(test_df['sentence'])
    test_wngrams_features  = vectorize_word.transform(test_df['sentence'])
    X_test = hstack([test_chngrams_features, test_wngrams_features])
    
    print 'Done Vectorize'
    
    # load model
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/trainDev_coarse_chngrams23456_ngrams123.clf', 'rb'))
    
    #clf = LogisticRegression(class_weight='balanced')
    #clf.fit(X, train_df['coarse'])
    # Save model
    # pickle.dump(clf, open('../data/DSL-training/task1/trainDev_coarse_chngrams23456_ngrams123.clf', 'wb'))
    
    print 'fit'
    predicted_y = clf.predict(X_test)
    
    if goldDA:
        y_test = test_df['coarse']
        print 'accuracy ' + str(accuracy_score(y_test, predicted_y))
        print 'f1-macro: ' + str(f1_score(y_test, predicted_y, average='macro'))
        
    output_file = codecs.open('../data/DSL-training/task1/coarseFineClassifiers/B2/B2_out_coarse_chngrams23456_ngrams123.tsv','w', encoding = 'utf8')
    if (goldDA):
        output_file.write('sentence\tpredicted\tgold\n')
        for item in zip(test_df['sentence'].tolist(), predicted_y, y_test):
            output_file.write(u"\t".join(item) + u"\n")
    else:
        output_file.write('sentence\tcoarse\n')
        for item in zip(test_df['sentence'].tolist(), predicted_y):
            output_file.write(u"\t".join(item) + u"\n")
    output_file.close()

if __name__ == "__main__":
    ngrams_LR(False)


# In[16]:

# create fine classifier per group
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv, codecs
from scipy.sparse import hstack 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
import pickle

vectorize_char =  CountVectorizer(tokenizer=None, analyzer = 'char', ngram_range=(2,6), 
                                lowercase = True)
vectorize_word =  CountVectorizer(tokenizer=None, analyzer = 'word', ngram_range=(1,3), 
                                lowercase = True)

#Create classifier per lang-group
def fine_classifier(train_df, test_df, outFile, goldDA, clf):
    train_chngrams_features = vectorize_char.fit_transform(train_df['sentence'])
    train_wngrams_features  = vectorize_word.fit_transform(train_df['sentence'])
    X = hstack([train_chngrams_features, train_wngrams_features])

    if len(test_df) > 0:
        test_chngrams_features = vectorize_char.transform(test_df['sentence'])
        test_wngrams_features  = vectorize_word.transform(test_df['sentence'])
        X_test = hstack([test_chngrams_features, test_wngrams_features])
        print 'DONE vectorize'

        #clf = LogisticRegression(class_weight='balanced')
        #clf.fit(X, train_df['fine'])

        # Save model
        #pickle.dump(clf, open(outFile + '.clf', 'wb'))

        print 'DONE TRAINING'
        predicted_y = clf.predict(X_test)
        print 'DONE PREDICTING'

        output_df = pd.DataFrame(test_df['idx'])
        output_df['sentence'] = test_df['sentence']
        output_df['pred_fine'] = predicted_y
        if goldDA:
            y_test = test_df['fine']
            output_df['fine'] = y_test
            print 'accuracy ' + str(accuracy_score(y_test, predicted_y))

        output_file = codecs.open(outFile,'w', encoding = 'utf8')
        if (goldDA):
            output_file.write('sentence\tpredicted\tgold\n')
            for item in zip(output_df['sentence'].tolist(), output_df['pred_fine'], output_df['fine']):
                output_file.write(u"\t".join(item) + u"\n")
        else:
            output_file.write('sentence\tpredicted\n')
            for item in zip(output_df['sentence'].tolist(), output_df['pred_fine']):
                output_file.write(u"\t".join(item) + u"\n")

        output_file.close()
        return output_df
    else:
        return []

def fine_per_group(outFile, goldDA):
    # Load the data
    train_file = "../data/DSL-training/task1/task1-trainDev-coarse_header.txt"
    test_file = "../data/DSL-training/task1/coarseFineClassifiers/B1/B1_out_coarse_chngrams23456_ngrams123.tsv"
    #test_file = "../data/DSL-training/task1/coarseFineClassifiers/B2/B2_out_coarse_chngrams23456_ngrams123.tsv"
    
    train_df = pd.read_csv(train_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    test_df = pd.read_csv(test_file, sep="\t", header = 0, encoding="utf8", quoting=csv.QUOTE_NONE)
    print train_df.shape
    train_df['idx'] = train_df.index
    test_df['idx']  = test_df.index + 1
    
    
    # Create dataframe per language group
    lang_groups = ['es', 'pt', 'fr', 'ss', 'ma']
    train_df_es = train_df.loc[train_df['coarse'] == 'es']
    test_df_es  = test_df.loc[test_df['coarse'] == 'es']
    
    train_df_pt = train_df.loc[train_df['coarse'] == 'pt']
    test_df_pt  = test_df.loc[test_df['coarse'] == 'pt']
    
    train_df_fr = train_df.loc[train_df['coarse'] == 'fr']
    test_df_fr  = test_df.loc[test_df['coarse'] == 'fr']
    
    train_df_ss = train_df.loc[train_df['coarse'] == 'ss']
    test_df_ss  = test_df.loc[test_df['coarse'] == 'ss']
    
    train_df_ma = train_df.loc[train_df['coarse'] == 'ma']
    test_df_ma  = test_df.loc[test_df['coarse'] == 'ma']
    
    #load model ES
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv.es.clf'))
    out_df_es = fine_classifier(train_df_es, test_df_es, outFile + '.es', goldDA, clf)
    print "DONE ES"
    #load model SS
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv.ss.clf'))
    out_df_ss = fine_classifier(train_df_ss, test_df_ss, outFile + '.ss', goldDA, clf)
    print "DONE SS"
    #load model PT
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv.pt.clf'))
    out_df_pt = fine_classifier(train_df_pt, test_df_pt, outFile + '.pt', goldDA, clf)
    print "DONE PT"
    #load model FR
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv.fr.clf'))
    out_df_fr = fine_classifier(train_df_fr, test_df_fr, outFile + '.fr', goldDA, clf)
    print "DONE FR"
    #load model MA
    clf = pickle.load(open('../data/DSL-training/task1/coarseFineClassifiers/models/out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv.ma.clf'))
    out_df_ma = fine_classifier(train_df_ma, test_df_ma, outFile + '.ma', goldDA, clf)
    print "DONE MA"
    mergeDataframes(out_df_es, out_df_ss, out_df_pt, out_df_fr, out_df_ma, outFile, goldDA)

# merge per index
def mergeDataframes(df1, df2, df3, df4, df5, outFile, goldDA):
    if len(df1) > 0:
        df1_2 = pd.concat([df1, df2], axis=0)
    else:
        df1_2 = df2
    if len(df4) > 0:
        df3_4 = pd.concat([df3, df4], axis=0)
    else:
        df3_4 = df3
    df1234 = pd.concat([df1_2, df3_4], axis=0)
    if len(df5) > 0:
        df = pd.concat([df1234, df5], axis=0)
    else:
        df = df1234
        
    df = df.sort_values(by='idx', ascending=1)
    output_file = codecs.open(outFile,'w', encoding = 'utf8')
    if (goldDA):
        output_file.write('sentence\tpredicted\tgold\n')
        for item in zip(df['sentence'].tolist(), df['pred_fine'], df['fine']):
            output_file.write(u"\t".join(item) + u"\n")
    else:
        output_file.write('sentence\tpredicted\n')
        for item in zip(df['sentence'].tolist(), df['pred_fine']):
            output_file.write(u"\t".join(item) + u"\n")
            
    output_file.close()
    return df

if __name__ == "__main__":
    #f = "../data/DSL-training/task1/coarseFineClassifiers/dev_out_coarseFine_goldCoarse_chngrams23456_ngrams123.tsv"
    f = "../data/DSL-training/task1/coarseFineClassifiers/B1/B1_chngrams23456_ngrams123_cascade.tsv"
    fine_per_group(f, False)
    


# In[ ]:



