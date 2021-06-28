#!/usr/bin/env python
# coding: utf-8

# ### Sentiment Analysis

# In[1]:


import pandas as pd
import numpy as np
import gzip

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')

# Datetime
from datetime import datetime

## Warnings
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("sample30.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# ### Data Cleaning

# In[5]:


#Select relevant features
df = df[["id",
"name",
"reviews_rating",
"reviews_text",
"reviews_title",
"reviews_username",
"user_sentiment"]
]


# In[6]:


#Missing vaue check

df.isnull().sum()


# In[7]:


df=df.dropna(subset=['reviews_title','reviews_username','user_sentiment'])
df.isnull().sum()


# In[8]:


df.shape


# In[9]:


#Concatenate reviews_title AND reviews_text
df['reviews'] = df[['reviews_title', 'reviews_text']].apply(lambda x: " ".join(str(y) for y in x if str(y) != 'nan'), axis = 1)
df1 = df.drop(['reviews_title', 'reviews_text'], axis = 1)
df1.head()


# In[10]:


#checking duplicates based on 'id', 'reviews_username' columns
df1[df1.duplicated(['id','reviews_username'],keep= False)]


# In[11]:


df1.shape


# In[12]:


df1 = df1.drop_duplicates(['id','reviews_username'], keep = 'first')
df1.shape


# In[13]:


#Rating distributon

plt.figure(figsize=(12,6))
df1['reviews_rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')


# In[14]:


#user_sentiment distributon

plt.figure(figsize=(6,5))
df1['user_sentiment'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')


# ### Text Preprocessing

# In[15]:


#To convert raw reviews to cleaned review
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import unicodedata
import contractions


# In[16]:


get_ipython().system('pip install contractions --user')


# In[17]:


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re, string, unicodedata
import nltk
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from contractions import contractions_dict


# In[18]:


# calculate raw tokens in order to measure of cleaned tokens

from nltk.tokenize import word_tokenize
raw_tokens=len([w for t in (df1["reviews"].apply(word_tokenize)) for w in t])
print('Number of raw tokens: {}'.format(raw_tokens))


# In[19]:


#Functions for Preprocessing

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text



# Define function to expand contractions
def expand_contractions(text):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match)                        if contractions_dict.get(match)                        else contractions_dict.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    
    try:
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
    except:
        return text
    return expanded_text
    


# special_characters removal
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation_and_splchars(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

stopword_list= stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopword_list:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    lemmas = lemmatize_verbs(words)
    return lemmas


# In[20]:


def normalize_and_lemmaize(input):
    sample = denoise_text(input)
    sample = expand_contractions(sample)
    sample = remove_special_characters(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)


# In[21]:


df1['clean_review'] = df1['reviews'].map(lambda text: normalize_and_lemmaize(text))
df1


# In[22]:


df1.head()


# In[23]:


clean_tokens=len([w for t in (df1["clean_review"].apply(word_tokenize)) for w in t])
print('Number of clean tokens: {}\n'.format(clean_tokens))
print('Percentage of removed tokens: {0:.2f}'.format(1-(clean_tokens/raw_tokens)))


# ### Exploratory Data Analysis

# In[ ]:


get_ipython().system('pip install pandas_profiling --user')


# In[25]:


import pandas_profiling
pandas_profiling.ProfileReport(df1)


# In[26]:


# Number of reviews for top 20 brands  

df2=pd.read_csv("sample30.csv")
brands = df2["brand"].value_counts()
plt.figure(figsize=(12,6))
brands[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 Brands")
plt.xlabel('Brand Name')
plt.ylabel('Number of Reviews')


# In[27]:


# Number of reviews for top 20 products  

products = df1["name"].value_counts()
plt.figure(figsize=(12,6))
products[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 Products")
plt.xlabel('Product Name')
plt.ylabel('Number of Reviews')


# In[28]:


# Number of reviews for bottom 20 products  

products = df1["name"].value_counts()
plt.figure(figsize=(12,6))
products[-20:].plot(kind='bar')
plt.title("Number of Reviews for Bottom 20 Products")
plt.xlabel('Product Name')
plt.ylabel('Number of Reviews')


# In[29]:


#Lower rating
df1[(df1['reviews_rating']<2)]


# In[30]:


from nltk.tokenize import RegexpTokenizer
def RegExpTokenizer(Sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(Sent)

ListWords = []
for m in df1['clean_review']:
    n = RegExpTokenizer(str(m))
    ListWords.append(n)
print(ListWords[10])


# In[31]:


get_ipython().system('pip install WordCloud --user')


# In[32]:


#All Words
from nltk import FreqDist
def Bag_Of_Words(ListWords1):
    all_words1 = []
    for m in ListWords1:
        for w in m:
            all_words1.append(w.lower())
    all_words2 = FreqDist(all_words1)
    return all_words2


# In[33]:


import matplotlib as mpl
from wordcloud import WordCloud
all_words = Bag_Of_Words(ListWords)
ax = plt.figure(figsize=(12,6))
# Generate a word cloud image
wordcloud = WordCloud(background_color='white',max_font_size=40).generate(' '.join(all_words.keys()))

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[34]:


#Review length vs Ratings
word_count=[]
for s1 in df1.clean_review:
    word_count.append(len(str(s1).split()))
    
plt.figure(figsize = (8,6))

sns.boxplot(x="reviews_rating",y=word_count,data=df1,showfliers=False)
plt.xlabel('Rating')
plt.ylabel('Review Length')

plt.show()


# In[35]:


# unique customers for each "rating class"
a = list(df.groupby(['user_sentiment'])['name'].unique())  

# number of customers
a2 = [len(a[0]),len(a[1])] 

# number of reviews for each "rating class"
b = list(df['user_sentiment'].value_counts())              

uniq_cust_rate = pd.DataFrame({'user_sentiment': ['Negative', 'Positive'],
                               'number_of_customers': a2,
                               'number_of_reviews': sorted(b)})
print(uniq_cust_rate)


# In[36]:


def length(text):
    length = len([w for w in nltk.word_tokenize(text)])
    return length

# Apply length function to create review length feature
df1['review_length'] = df1['clean_review'].apply(length)


# In[37]:


df1.head(2)


# In[38]:


sns.pairplot(df1)


# ### Best-suited vectorizer & Modeling

# In[ ]:


get_ipython().system('pip install spacy --user')
get_ipython().system('pip install gensim --user')
get_ipython().system('pip install catboost --user')


# In[43]:


get_ipython().system('pip install -U pip setuptools wheel')
get_ipython().system('pip install -U spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[49]:


import unicodedata
tokenizer = nltk.tokenize.toktok.ToktokTokenizer()

import spacy
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

## Modeling
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier, Pool
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from gensim.models import Word2Vec
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier


# In[51]:


df1['rating_class'] = df1['user_sentiment'].apply(lambda x: 0 if x == 'Negative' else 1)

# Splitting the Data Set into Train and Test Sets
X = df1['clean_review']
y = df1['rating_class']


# In[156]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Print train and test set shape
print ('Train Set Shape\t\t:{}\nTest Set Shape\t\t:{}'.format(X_train.shape, X_test.shape))


# In[70]:


from itertools import product


# In[90]:


#Confusion Matrix Plot Function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.ocean):
    """
    Create a confusion matrix plot for 'Positive' and 'Negative' rating values 
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 15)
    plt.yticks(tick_marks, classes, fontsize = 15)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", 
                 color = "white" if cm[i, j] < thresh else "black", fontsize = 30)
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize = 20)
    plt.xlabel('Predicted Label', fontsize = 20)

    return plt

def disp_confusion_matrix(y_pred, model_name, vector = 'CounterVectorizing'):
    """
    Display confusion matrix for selected model with countVectorizer
    """
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8, 8))
    plot = plot_confusion_matrix(cm, classes=['Negative','Positive'], normalize=False, 
                                 title = model_name + " " + 'with' + " " + vector + " "+ '\nConfusion Matrix')
    plt.show()


# ### Bag of Words

# In[54]:


#Using CountVectorizer Bag of Words
# Create the word vector with CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,1))
count_vect_train = count_vect.fit_transform(X_train)
count_vect_train = count_vect_train.toarray()
count_vect_test = count_vect.transform(X_test)
count_vect_test = count_vect_test.toarray()


# In[55]:


# Print vocabulary length
print('Vocabulary length :', len(count_vect.get_feature_names()))


# In[56]:


# Assign feature names of vector into a variable
vocab = count_vect.get_feature_names()

# Dataframe for train countvectorizer dataset
pd.DataFrame(count_vect_train, columns = vocab).head()


# In[57]:


#Function for applying differet algorithm
def modeling(Model, Xtrain = count_vect_train, Xtest = count_vect_test):
        
    # Instantiate the classifier: model
    model = Model
    
    # Fitting classifier to the Training set (all features)
    model.fit(Xtrain, y_train)
    
    global y_pred
    # Predicting the Test set results
    y_pred = model.predict(Xtest)
    
    # Assign f1 score to a variable
    score = f1_score(y_test, y_pred, average = 'weighted')
    
    # Printing evaluation metric (f1-score) 
    print("f1 score: {}".format(score))



# ### 4. XGBoost

# In[158]:


# Call the modeling function for XGBoost with countvectorizer and print f1 score
modeling(XGBClassifier())
# Assign y_pred to a variable for further process
y_pred_cv_xgb = y_pred


# In[67]:


# Compute and print the classification report
print(classification_report(y_test, y_pred_cv_xgb))


# In[94]:


# Print confusion matrix for gradient boosting with countVectorizer
disp_confusion_matrix(y_pred_cv_xgb, "XGBoost")


# #### Best Model XGBoost 
# #### on the basis of highest average f1 score

# ## XXXX

# ### Recommendation System


# ### Item Based

# Taking the transpose of the rating matrix to normalize the rating around the mean for different product.

# In[267]:


df_pivot = train.pivot(
    index='user_name',
    columns='product_name',
    values='rating'
).T

df_pivot.head()


# Normalising the product rating for each product for using the Adujsted Cosine

# In[268]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[269]:


df_subtracted.head()


# Finding the cosine similarity using pairwise distances approach

# In[270]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# Filtering the correlation only for which the value is greater than 0. (Positively correlated)

# In[271]:


item_correlation[item_correlation<0]=0
item_correlation


# In[272]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[273]:


item_predicted_ratings.shape


# In[274]:


dummy_train.shape


# In[275]:


#Filtering the rating only for the products not rated by the user for recommendation
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# #### Finding the top 20 recommendation for the *user*

# In[312]:


# Take the user ID as input
user_input = input("Enter your user name")
print(user_input)


# In[313]:


# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# #### Evaluation of item based

# In[284]:


#Evaluation for the product already rated by the user insead of predicting it for the product not rated by the user
test.columns


# In[285]:


common =  test[test.product_name.isin(train.product_name)]
common.shape


# In[286]:


common.head(4)


# In[287]:


common_item_based_matrix = common.pivot_table(index='user_name', columns='product_name', values='rating').T


# In[288]:


common_item_based_matrix.shape


# In[289]:


item_correlation_df = pd.DataFrame(item_correlation)


# In[290]:


item_correlation_df.head(1)


# In[291]:


item_correlation_df['product_name'] = df_subtracted.index
item_correlation_df.set_index('product_name',inplace=True)
item_correlation_df.head()


# In[292]:


list_name = common.product_name.tolist()


# In[293]:


item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[294]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T


# In[295]:


item_correlation_df_3.head()


# In[296]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[297]:


common_item_predicted_ratings.shape


# In[298]:


dummy_test = common.copy()

dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='user_name', columns='product_name', values='rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# In[299]:


common_ = common.pivot_table(index='user_name', columns='product_name', values='rating').T


# In[300]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[301]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[302]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### Selecting *Item based Recommendation System* 
# #### because there are many users who have rated only 1 product so it's difficult to find recommendation. In Item based, the products have been rated by multiple users so it's easy to find similar recommendation. The evaluation matric is almost same.

# ### Predicting the sentiment (positive or negative) of all the reviews for top 20 recommended products for a user

# In[335]:


df3 = df1[df1['name'].isin(pd.DataFrame(d).index)]

count_vect_f = count_vect.transform(df3.clean_review)
count_vect_f = count_vect_f.toarray()

#Best model selected XGBoost
model=XGBClassifier()
model.fit(count_vect_train, y_train)

pred_f = model.predict(count_vect_f)
pred_f


# In[390]:


import pickle
pickle.dump(model, open('xgb_model.pkl', 'wb'))


# In[348]:


pred_f_df = pd.DataFrame(pred_f, columns=['user_sentiment'])
pred_f_df = pd.concat([pred_f_df, df3['name'].reset_index(drop=True)], axis=1)
pred_f_df


# ### % of positive sentiments for all the reviews of of top 20 products

# In[376]:


p = round((pred_f_df.groupby(['name'])['user_sentiment'].sum()/pred_f_df.groupby(['name'])['user_sentiment'].count())*100,1)
p


# ### Top 5 products with the highest % of positive reviews

# In[388]:


top_5_final = pd.DataFrame(p).sort_values('user_sentiment', ascending=False).reset_index().iloc[0:5,0]
top_5_final

