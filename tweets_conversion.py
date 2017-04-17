
# coding: utf-8


# Ayah Zirikly --varDial shared task (ar)
# Tweets normalization and buckwalter conversion

# In[2]:

import re, codecs, itertools

from collections import OrderedDict

countryCode2DA = {"AE": "GLF", 
                 "BH": "GLF", 
                 "EG": "EGY", 
                 "IL": "LAV", 
                 "IQ": "GLF", 
                 "JO": "LAV", 
                 "KW": "GLF", 
                 "LB": "LAV", 
                 "MA": "NOR", 
                 "MR": "NOR", 
                 "OM": "GLF", 
                 "QA": "GLF", 
                 "SA": "GLF", 
                 "SY": "LAV", 
                 "TN": "NOR",
                 "DZ": "NOR",
                 }

buck2uni = {"'": u"\u0621", # hamza-on-the-line
            "|": u"\u0622", # madda
            ">": u"\u0623", # hamza-on-'alif
            "&": u"\u0624", # hamza-on-waaw
            "<": u"\u0625", # hamza-under-'alif
            "}": u"\u0626", # hamza-on-yaa'
            "A": u"\u0627", # bare 'alif
            "b": u"\u0628", # baa'
            "p": u"\u0629", # taa' marbuuTa
            "t": u"\u062A", # taa'
            "v": u"\u062B", # thaa'
            "j": u"\u062C", # jiim
            "H": u"\u062D", # Haa'
            "x": u"\u062E", # khaa'
            "d": u"\u062F", # daal
            "*": u"\u0630", # dhaal
            "r": u"\u0631", # raa'
            "z": u"\u0632", # zaay
            "s": u"\u0633", # siin
            "$": u"\u0634", # shiin
            "S": u"\u0635", # Saad
            "D": u"\u0636", # Daad
            "T": u"\u0637", # Taa'
            "Z": u"\u0638", # Zaa' (DHaa')
            "E": u"\u0639", # cayn
            "g": u"\u063A", # ghayn
            "_": u"\u0640", # taTwiil
            "f": u"\u0641", # faa'
            "q": u"\u0642", # qaaf
            "k": u"\u0643", # kaaf
            "l": u"\u0644", # laam
            "m": u"\u0645", # miim
            "n": u"\u0646", # nuun
            "h": u"\u0647", # haa'
            "w": u"\u0648", # waaw
            "Y": u"\u0649", # 'alif maqSuura
            "y": u"\u064A", # yaa'
            "F": u"\u064B", # fatHatayn
            "N": u"\u064C", # Dammatayn
            "K": u"\u064D", # kasratayn
            "a": u"\u064E", # fatHa
            "u": u"\u064F", # Damma
            "i": u"\u0650", # kasra
            "~": u"\u0651", # shaddah
            "o": u"\u0652", # sukuun
            "`": u"\u0670", # dagger 'alif
            "{": u"\u0671", # waSla
}

uni2buck={}
# Iterate through all the items in the buck2uni dict.
for (key, value) in buck2uni.iteritems():
    # The value from buck2uni becomes a key in uni2buck, and vice versa for the keys.
    uni2buck[value] = key
        
def buckwalter2 (text):
    buckArab = {"'":"ء", "|":"آ", "?":"أ", "&":"ؤ", "<":"إ", "}":"ئ", "A":"ا", "b":"ب", "p":"ة", "t":"ت", "v":"ث", "g":"ج", "H":"ح", "x":"خ", "d":"د", "*":"ذ", "r":"ر", "z":"ز", "s":"س", "$":"ش", "S":"ص", "D":"ض", "T":"ط", "Z":"ظ", "E":"ع", "G":"غ", "_":"ـ", "f":"ف", "q":"ق", "k":"ك", "l":"ل", "m":"م", "n":"ن", "h":"ه", "w":"و", "Y":"ى", "y":"ي", "F":"ً", "N":"ٌ", "K":"ٍ", "~":"ّ", "o":"ْ", "u":"ُ", "a":"َ", "i":"ِ"}
    ordbuckArab = {ord(v.decode('utf8')): unicode(k) for (k, v) in buckArab.iteritems()}

def transliterateString(inString):

	out = ""

	# Loop over each character in the string, inString.
	for char in inString:
		# Look up current char in the dictionary to get its
		# respective value. If there is no match, e.g., chars like
		# spaces, then just stick with the current char without any
		# conversion.
		out = out + uni2buck.get(char, char)

	return out

def bw_file_hmubarakFormat(inFile, DA_code):
    with codecs.open(inFile, 'r', encoding='utf8') as f, codecs.open(inFile + '.bw', 'w', encoding='ascii', errors='ignore') as out:
        lines = f.readlines()
        for line in lines:
            words = line.rstrip().split(' ')
            result = transliterateString(words[0])
            result = replace_hyperlinks(result)
            result =  re.sub(r"@\S+", "", result)
            result = re.sub('\s+', ' ', result).strip()
            out.write(elongatedWords(result))
            for i in range(1, len(words)):
                result = transliterateString(words[i])
                result = replace_hyperlinks(result)
                result =  re.sub(r"@\S+", "", result)
                result = re.sub('\s+', ' ', result).strip()
                out.write(' ' + '%s' % elongatedWords(result))
            #pattern = re.compile('|'.join(uni2buck.keys()))
            #result = pattern.sub(lambda x: uni2buck[x.group()], line)
            #print result
            out.write('\t' + DA_code + '\n')
            
def bw_file_ayaFormat(inFile):
    with codecs.open(inFile, 'r', encoding='utf8') as f,             codecs.open(inFile[:inFile.rfind('.')] + '.bw.norm.tsv', 'w', encoding='ascii', errors='ignore') as out:
        out.write('sentence\tDA\n')
        lines = f.readlines()
        firstLine = True
        for line in lines:
            if firstLine:
                firstLine = False
                continue
            textCodeLoc = line.rstrip().split('\t')
            if len(textCodeLoc) == 4:
                if (textCodeLoc[1] in countryCode2DA.keys()):
                    pattern_code = re.compile('|'.join(countryCode2DA.keys()))
                    da = pattern_code.sub(lambda x: countryCode2DA[x.group()], textCodeLoc[1])
                    pattern_text = re.compile('|'.join(uni2buck.keys()))
                    text = pattern_text.sub(lambda x: uni2buck[x.group()], textCodeLoc[3])
                    #text_noEmoji = remove_emoji3(text)
                    #text_noEmoji = ''.join(c for c in text if c <= '\uFFFF')
                    #text_normURL = re.sub(r"http\S+", "", text_noEmoji)
                    text_normURL = replace_hyperlinks(text)
                    text = re.sub(r"@\S+", "", text_normURL)
                    text = re.sub('\s+', ' ', text).strip()
                    #"".join(OrderedDict.fromkeys(text))
                    out.write(elongatedWords(text) + '\t' + da + '\n')

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

def elongatedWords(s):
    return re.sub(r'(.)\1+', r'\1\1', s)
    #return  re.sub(ur"([^0-9\.])(?i)(\1{2,})", r"\1\1", s,10)

def hMubarakFiles(path):
    f = path + "DialectTweets-Sample1000/DialectTweetsAE-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsBH-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsDZ-Sample.txt";
    bw_file_hmubarakFormat(f, "NOR")
    f = path + "DialectTweets-Sample1000/DialectTweetsEG-Sample.txt";
    bw_file_hmubarakFormat(f, "EGY")
    f = path + "DialectTweets-Sample1000/DialectTweetsJO-Sample.txt";
    bw_file_hmubarakFormat(f, "LAV")
    f = path + "DialectTweets-Sample1000/DialectTweetsKW-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsLB-Sample.txt";
    bw_file_hmubarakFormat(f, "LAV")
    f = path + "DialectTweets-Sample1000/DialectTweetsMA-Sample.txt";
    bw_file_hmubarakFormat(f, "NOR")
    f = path + "DialectTweets-Sample1000/DialectTweetsOM-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsPL-Sample.txt";
    bw_file_hmubarakFormat(f, "LAV")
    f = path + "DialectTweets-Sample1000/DialectTweetsQA-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsSA-Sample.txt";
    bw_file_hmubarakFormat(f, "GLF")
    f = path + "DialectTweets-Sample1000/DialectTweetsSY-Sample.txt";
    bw_file_hmubarakFormat(f, "LAV")
    f = path + "DialectTweets-Sample1000/DialectTweetsTN-Sample.txt";
    bw_file_hmubarakFormat(f, "NOR")
    
if __name__ == "__main__":
    path = '/Users/ayousha/nlp/varDial_sharedTask_coling2016/data/DSL-training/ar/resources/'
    #hMubarakFiles(path)
    #f = path + "DA_tweets_0828.tsv"
    bw_file_ayaFormat(f)


# In[ ]:



