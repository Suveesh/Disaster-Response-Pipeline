#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[ ]:


# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')


import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer


# In[ ]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql("SELECT * FROM Disaster", engine)
X = df['message']
y = df.iloc[:,1:]

# ### 2. Write a tokenization function to process your text data

# In[ ]:


def tokenize(text):
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()


    # Remove stop words
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='n').strip()
        #I passed in the output from the previous noun lemmatization step. This way of chaining procedures is very common.
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        #It is common to apply both, lemmatization first, and then stemming.
        clean_tok =PorterStemmer().stem(clean_tok)
        
        clean_tokens.append(clean_tok)
    
    
    return clean_tokens
    


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[ ]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4],
    }

# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


pipeline.fit(X_train, y_train)

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:

target_names = y.columns
print(classification_report(y_test, y_pred, target_names=target_names))







# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


#parameters = 

#cv = GridSearchCV(pipeline, param_grid=parameters)

# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:
#cv.fit(X_train, y_train)

#y_pred = cv.predict(x_train)


#print(classification_report(y_test, y_pred, target_names=target_names))



# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]: 





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




