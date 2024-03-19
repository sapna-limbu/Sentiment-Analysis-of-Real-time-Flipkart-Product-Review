#!/usr/bin/env python
# coding: utf-8

# ## Objective
# The objective of this project is to classify customer reviews as positive or negative and understand the pain points of customers who write negative reviews. By analyzing the sentiment of reviews, we aim to gain insights into product features that contribute to customer satisfaction or dissatisfaction.
# 
# 

# In[1]:


# Importing required libraries
import numpy as np
import pandas as pd
import re
import emoji
import autocorrect ## Has to be installed
from textblob import TextBlob ## Has to be installed
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
## nltk.download('punkt')
from nltk.corpus import stopwords
## nltk.download('stopwords')
from nltk.stem import PorterStemmer,WordNetLemmatizer,LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from warnings import filterwarnings
filterwarnings("ignore")
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,ConfusionMatrixDisplay


# In[2]:


#Loading the dataset
df = pd.read_csv("data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# ## Text Preprocessing

# In[5]:


# Checking Missing values
df.isna().sum()


# In[6]:


df['Month'].fillna(df['Month'].mode()[0], inplace=True)
df['Place of Review'].fillna(df['Place of Review'].mode()[0], inplace=True)
df['Down Votes'].fillna(df['Down Votes'].mean(), inplace=True)
df['Up Votes'].fillna(df['Up Votes'].mean(), inplace=True)


# In[7]:


# Drop rows with NaN values in the 'Review text' column
df.dropna(subset=['Review text'],inplace=True)
df.dropna(subset=['Reviewer Name'],inplace=True)
df.dropna(subset=['Review Title'],inplace=True)


# In[8]:


df.isna().sum()


# In[9]:


# Checking duplicates 
df.duplicated().sum()


# In[10]:


#Converting to lower case
df['Review text']=df['Review text'].str.lower()


# In[11]:


df['Review text'].head()


# In[ ]:





# In[12]:


import emoji

# Define a function to remove emojis from a text
def remove_emojis(text):
    return emoji.demojize(text)

# Apply the function to the 'Review text' column
df['Review text'] = df['Review text'].apply(remove_emojis)




# In[ ]:





# In[13]:


# Define a function to convert text to lowercase
def convert_to_lower(text):
    return text.lower()

# Apply the function to the 'Review text' column
df['Review text'] = df['Review text'].apply(convert_to_lower)


# In[14]:


df['Review text'].head(2)


# ## Tokenization

# In[15]:


df['Review text'] = df['Review text'].apply(lambda x: " ".join(word_tokenize(x)))


# In[16]:


df.head(2)


# ## Removing Punctuation marks and Numbers

# In[17]:


import pandas as pd
import re

def remove_punctuation_and_numbers(text):
    # Remove punctuation and numbers using regular expressions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    return text

# Assuming df is your DataFrame and 'Review text' is the column name
df['Review text'] = df['Review text'].apply(lambda x: remove_punctuation_and_numbers(x))



# In[18]:


df


# ## Removing the Stopwords

# In[19]:


sw = stopwords.words('english')


# In[20]:


sw


# In[21]:


df['Review text'] = df['Review text'].str.replace('â','a')
df['Review text'] = df['Review text'].str.replace('¹','')


# In[22]:


df['Review text']


# In[23]:


def remove_stopwords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in sw]
    
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

# Assuming df is your DataFrame and 'Review text' is the column name
df['Review text'] = df['Review text'].apply(remove_stopwords)



# In[24]:


df['Review text']


# ## Text Normalization

# In[25]:


lemma=WordNetLemmatizer()


# In[26]:


from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

def lemmatizer(text):
    lemmatized_words = [lemma.lemmatize(word) for word in text.split()]
    return " ".join(lemmatized_words)

# Assuming yonex is your DataFrame and 'Review text' is the column name
df['Review text'] = df['Review text'].apply(lemmatizer)


# In[27]:


df['Review text']


# In[28]:


df


# ## Numerical Feature Extraction

# In[29]:


vector = CountVectorizer()


# In[30]:


vector.fit_transform(df['Review text']).toarray()


# In[31]:


text = "".join(df['Review text'].values.tolist())
data=WordCloud().generate(text)
plt.imshow(data)


# In[32]:


SA = SentimentIntensityAnalyzer()


# In[33]:


def polarity(x):
    return SA.polarity_scores(x)['compound']


# In[34]:


df['Emotion'] = df['Review text'].apply(polarity)


# In[35]:


def sentiment(x):
    if x > 0.4:
        return 'Positive'
    
    elif x < 0.4:
        return 'Negative'
    
    elif x > 0.2 and x < 0.4:
        return 'Neutral'


# In[36]:


df['Emotion'] = df['Emotion'].apply(sentiment)


# In[37]:


import matplotlib.pyplot as plt

# Assuming yonex is your DataFrame and 'Emotion' is the column name
df['Emotion'].value_counts().plot(kind='barh', color='orange')  # Horizontal bar plot
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.title('Emotion Distribution')
plt.show()


# In[38]:


data = pd.concat([df['Review text'],df['Emotion']],axis=1)


# In[39]:


data.head(5)


# In[40]:


df['Emotion'].value_counts()


# ## Identify Input and Output

# In[41]:


X=df['Review text']
y=df['Emotion']


# In[42]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[43]:


pipe = Pipeline([('Vectorization',CountVectorizer(stop_words='english')),
              ("Estimator",MultinomialNB(alpha=3))])


# In[44]:


pipe.fit(X_train,y_train)


# In[45]:


y_pred = pipe.predict(X_test)


# In[46]:


accuracy_score(y_test,y_pred)


# In[47]:


f1_score(y_test, y_pred, average='weighted')


# ## Creating a pickle file

# In[48]:


import pickle


# In[49]:


pickle.dump(pipe,open("sentiment.pkl",'wb'))


# In[50]:


import os


# In[51]:


os.getcwd()


# In[52]:


predict = pickle.load(open("sentiment.pkl",'rb'))


# In[53]:


predict.predict([" very nice product"])


# In[54]:


predict.predict(["such a bad product please don't buy"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




