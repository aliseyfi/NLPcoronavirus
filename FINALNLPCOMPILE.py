#!/usr/bin/env python
# coding: utf-8

# In[3]:


#LOADING THE MODULES
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
#from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tweepy
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import csv
import re #regular expression
import textblob
from textblob import TextBlob
import string
import preprocessor as p
import nltk
import nltk.data
nltk.download('punkt')
nltk.download('stopwords')
from os import listdir
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from os.path import isfile, join
from nltk.util import bigrams 
from nltk.corpus import stopwords 
from nltk.tokenize import TreebankWordTokenizer
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
treebank_tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english')) 


# In[11]:


#LOADING AND COMBINING THE SETS
df17=pd.read_csv('March17')
df18=pd.read_csv('March18')
df19=pd.read_csv('March19')
df17['Date']='March17'
df18['Date']='March18'
df19['Date']='March19'
df= df17.append(df18, ignore_index=True)
df= df.append(df19, ignore_index=True)
#cases_by_state=pd.read_csv('Coronavirus17')
#Labeled=pd.read_csv('Labelled_data')


# In[12]:


#CLEANING THE TWEETS
df=df.drop(['source','status_id','user_id','screen_name','reply_to_status_id','reply_to_user_id','reply_to_screen_name','place_type','friends_count','account_created_at','account_lang','lang','is_quote','is_retweet'],axis=1)
df= df[df['country_code'].notna()]
df= df[df['country_code'].str.contains('US')] 
def determine_state(place_full_name):
    return place_full_name[-2:]
df['State']=df['place_full_name'].apply(determine_state)
df=df.drop(['country_code'],axis=1)
df['text']=df['text'].str.lower()
import re,string
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
df['text']=df['text'].apply(strip_links)
df['text']=df['text'].apply(strip_all_entities)
df=df.drop(['created_at'],axis=1)
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
emojipatterns = re.compile("["
         u"\U0001F600-\U0001F64F"  
         u"\U0001F300-\U0001F5FF"  
         u"\U0001F680-\U0001F6FF" 
         u"\U0001F1E0-\U0001F1FF"  
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
emoticons = emoticons_happy.union(emoticons_sad)
def clean_tweets(tweet):
 
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = emojipatterns.sub(r'', tweet)
    tweet = word_tokenize(tweet)
    st = PorterStemmer()
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    for w in word_tokens:
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
df['filtered']=df['text'].apply(clean_tweets)
stem = PorterStemmer()
df['filtered'] = df['filtered'].apply(lambda x: " ".join([stem.stem(word) for word in x.split()]))
df['clean_text']=df['text'].apply(clean_tweets)


# In[15]:


#SENTIMENT ANALYSIS AND DROPPING NON EMOTIONAL TWEETS 
import seaborn as sns
def sentiment_category(x):
    if 0>x:
        return 'negative'
    elif x==0:
        return 'neutral'
    elif x>0:
        return 'positive'
    
def subjectivity_category(x):
    if 0<=x<=0.5:
        return 'objective'
    elif x>0.5:
        return 'subjective'
df['polarity']=0
df['subjectivity']=0
df['sentcat']=df['polarity'].apply(sentiment_category)
df['subjcat']=df['subjectivity'].apply(subjectivity_category)
df= df[df['State'].isin(['CA','NY','TX','FL','IL','OH','MA','PA','SA','PA','WA'])]
sns.barplot(x='sentcat', y='polarity', hue='State', data=df,saturation=0.8)
df['subjcat'].value_counts().plot(kind='bar')
df['sentcat'].value_counts().plot(kind='bar')
df= df[df['sentcat'].isin(['positive','negative'])]


# In[26]:


#BUILDING THE CLASSIFIERS
#ENCODING THE LABELS
le = LabelEncoder()
filtered["emotion_cat"] = le.fit_transform(labeled["emotions"])
#CONV EN LISTE ET FIT / MAX FEATURES
tfidf=TfidfVectorizer()
tfidfconverter = TfidfVectorizer(max_features=30000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))  
labeled['transformed_tweet']=tfidf.fit_transfrorm(df['filtered'])
myset=labeled[['emotions','transformed_tweet']].copy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# OUBLIE PAS DE TIME IT 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
classifier1 = DecisionTreeRegressor()
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, classifier1.predict(X_test)))
classifier2 = GaussianNB()
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm) 
print(classification_report(y_test, classifier2.predict(X_test)))
classifier3 = LinearSVC()
classifier3.fit(X_train, y_train)
y_pred = classifier3.predict(X_test)
cm = confusion_matrix(y_test, y_pred
print(classification_report(y_test, classifier3.predict(X_test)))
classifier4 = SVC(kernel='linear') 
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)
cm = confusion_matrix(y_test, y_pred
print(classification_report(y_test, classifier4.predict(X_test)))
classifier5 = SVC(gamma='scale')
classifier5.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)
cm = confusion_matrix(y_test, y_pred
print(classification_report(y_test, classifier5.predict(X_test)))
classifier6 = LogisticRegression()
classifier6.fit(X_train, y_train)
y_pred = classifier6.predict(X_test)
cm = confusion_matrix(y_test, y_pred
print(classification_report(y_test, classifier6.predict(X_test)))
classifier7 = KNeighborsClassifier(n_neighbors=4) #mieux que 5
classifier7.fit(X_train, y_train)
y_pred = classifier7.predict(X_test)
cm = confusion_matrix(y_test, y_pred
print(classification_report(y_test, classifier7.predict(X_test)))
print(cross_val_score(classifier1, X_train, y_train, cv=10))  
print(cross_val_score(classifier2, X_train, y_train, cv=10))  
print(cross_val_score(classifier3, X_train, y_train, cv=10))  
print(cross_val_score(classifier4, X_train, y_train, cv=10))  
print(cross_val_score(classifier5, X_train, y_train, cv=10))  
print(cross_val_score(classifier6, X_train, y_train, cv=10))  
print(cross_val_score(classifier7, X_train, y_train, cv=10))  
predicted_emotions=classifier2.predict(Xtest)                     
df['predicted_emotion']=predicted_emotions
mydict=cases_by_state.to_dict()
#df['number_cases'] ajuster moi meme excel
myfile=df.to_csv('Addcases') 
mycomplete_file=pd.read_csv('finalfile')
finalfile.head()
mymodel= glm(formula='rate~ friends_count+C(predicted_emotions)+C(state)+C(Date)', data=finalfile, family=sm.families.Binomial()).fit()# bd


# In[ ]:




