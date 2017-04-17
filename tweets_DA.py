'''
Ayah Zirikly --varDial shared task
Collect twitter data per country to add to the training data
'''

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import csv, json, sys, codecs, re

access_token = '1473130034-P4tMn2YioXQpMhBXrxUrvxOPANw2aAP70U6lLuA'
access_secret = 'PnxBLOC1TpSGPQmbMecvWWG7olbGgNEE60KQPH3e5TcE0'
consumer_key = 'KHm2w0FUIPr4JLD97czv5DJYn'
consumer_secret = 'RHMt3P2deaZ4q9JAuTNoN6VhUkrVoUYrm5okq76AGWl3t19DYy'


csvfile = codecs.open('../varDial_sharedTask_coling2016/data/DSL-training/ar/resources/DA_tweets.tsv','wb', encoding='utf8')

class StdOutListener(StreamListener):
    
    def on_data(self, data):
        tweet = json.loads(data)
        
        if (tweet.get('place')):
            print '%s\t%s\t%s\t%s' % (tweet['id'], tweet['text'].encode('utf8'), tweet['place']['country'].encode('utf8'), tweet['place']['country_code'].encode('utf8'))
            
            row = u"\t".join([unicode(tweet['id']),
                              unicode(tweet['place']['country_code']),
                              tweet['place']['country'],
                              tweet['text'].replace('\n', '')])
            csvfile.write(row + '\n')
        
        return True
    
    def on_error(self, status):
        print status


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    csvfile.write('ID\tcountryCode\tcountry\ttext\n')
    stream = Stream(auth, l)
    #tweets = stream.filter(locations = [13,5,37,50.861], languages=["ar"])
    tweets = stream.filter(locations = [-16.8,13.3,59.3,36.1], languages=['ar'])
