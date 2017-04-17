
# coding: utf-8
# Ayah Zirikly --varDial shared task
# Test set B1 and B2 preprocessing (normalization)

# In[21]:


import re, codecs, itertools
from collections import OrderedDict
 
html_escape_table = {
        "&amp;": "&",
        "&quot;": '"',
        "&apos;": "'",
        "&gt;": ">",
        "&lt;": "<",
        }

def normalizeTweets(inFile):
    with codecs.open(inFile, 'r', encoding='utf8') as f,             codecs.open(inFile[:inFile.rfind('.')] + '.norm.txt', 'w', encoding='utf8', errors='ignore') as out:
        lines = f.readlines()
        for line in lines:
            firstTab = True
            texts = line.rstrip().split('\t')
            for text in texts:
                text = remove_emoji(text)
                text = remove_emoji2(text)
                text_normURL = remove_hyperlinks(text)
                text = re.sub(r"@\S+", "", text_normURL)
                text = re.sub(r"#\S+", "", text)
                text = re.sub(r"RT ", "", text)
                # remove HTML tags (e.g. &lt)
                pattern = re.compile('|'.join(html_escape_table.keys()))
                text = pattern.sub(lambda x: html_escape_table[x.group()], text)
                text = re.sub('\s+', ' ', text).strip()
                if (not firstTab):
                    out.write(u"\t" + unicode(elongatedWords(text)))
                else:
                    out.write(unicode(elongatedWords(text)))
                firstTab = False
            out.write(u"\n")
            
def remove_emoji(text):
    if not text:
        return text
    if not isinstance(text, basestring):
        return text
    try:
    # UCS-4
        patt = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        patt = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return patt.sub('', text)

def remove_emoji2(text):
    try:
        # Wide UCS-4 build
        myre = re.compile(u'['
            u'\U0001F914'
            u'\U0001F300-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+', 
            re.UNICODE)
    except re.error:
        # Narrow UCS-2 build
        myre = re.compile(u'('
            u'\1F914'
            u'\ud83c[\udf00-\udfff]|'
            u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
            u'[\u2600-\u26FF\u2700-\u27BF])+', 
            re.UNICODE)
    return myre.sub('', text)

email=re.compile(r'[^ ]+\@[^ ]+\.[^ \n]+')
hyp1=re.compile(r"http:/\S+")
hyp2=re.compile(r"www\.\S+")
hyp3=re.compile(r"https:/\S+")

def replace_hyperlinks(s):
    s = hyp1.sub(u"url", s)
    s = hyp2.sub(u"url", s)
    s = hyp3.sub(u"url", s)
    s = email.sub(u"url", s)
    return s

def remove_hyperlinks(s):
    s = hyp1.sub(u"", s)
    s = hyp2.sub(u"", s)
    s = hyp3.sub(u"", s)
    s = email.sub(u"", s)
    return s

def elongatedWords(s):
    return re.sub(r'(.)\1+', r'\1\1', s)

if __name__ == "__main__":
    normalizeTweets('../data/DSL2016-test/B1.txt')
    normalizeTweets('../data/DSL2016-test/B2.txt')



# In[ ]:



