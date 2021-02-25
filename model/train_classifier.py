# In[1]:

import sys
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
from sklearn.model_selection import GridSearchCV
import pickle

# In[2]:
def load_data(database_filepath):

    '''
    Function to retreive data from sql database (database_filepath) and split the dataframe into X and y variable

    Input: Databased filepath
    Output: Returns the Features X & target y along with target columns names catgeory_names

    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    X = df['message']
    y = df.iloc[:,4:]
# In[3]:

def tokenize(text):
    '''
    Function to clean the text data  and apply tokenize and lemmatizer function
    Return the clean tokens

    Input: text
    Output: cleaned tokenized text as a list object
    '''

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
        
        clean_tokens.append(clean_tok)
    
    
    return clean_tokens
# In[4]:

def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv

    Input: N/A
    Output: Returns the model
    '''



    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {'tfidf__norm': ['l1','l2'],
              'clf__estimator__criterion': ["gini", "entropy"]    
             }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

       

    

# In[5]:
def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Function to evaluate a model and return the classificatio and accurancy score.

    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score

    '''

    y_pred = model.predict(X_test)
    report= classification_report(Y_test,y_pred, target_names=category_names)


    temp=[]
    for item in report.split("\n"):
        temp.append(item.strip().split('     '))
    clean_list=[ele for ele in temp if ele != ['']]
    report_df=pd.DataFrame(clean_list[1:],columns=['group','precision','recall', 'f1-score','support'])


    return report_df
    



# In[6]:


def save_model(model, model_filepath):

    '''
    Function to save the model as pickle file in the directory

    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 

    '''

    with open('model_filepath', 'wb') as file:
        pickle.dump(model, file)

# In[7]:

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

# In[8]:
if __name__ == '__main__':
    main()
# %%

# %%
