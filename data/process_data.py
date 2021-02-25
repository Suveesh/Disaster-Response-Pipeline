# In[1]:
import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

# In[2]:
def load_data(messages_filepath, categories_filepath):

    ''''

    Function to load Messages and Categories Data set from the csv file and merge  
    into a single data frame named df variable

    return a dataframe is merged on id column in both messages and categories data frame

    Input: messages_filepath, categories_filepath
    Output: Merged dataframe of messages and categories dataframe
    
    ''''
   #Read csv file and load in the variable as dataframe

    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    #merge message_df and categories_df on id column
    df = pd.merge(messages_df, categories_df, on="id", how='inner')

    return df

# In[3]:
def clean_data(df):

    ''''

    Function to clean the dataframe inorder to be compatible for the machinelearning application.

    Split the categories column with delimit ';' and
    Convert the first row values in categories dataframe to the column headers. 

    Convert the numerical values other than 0 and 1 as 1.

    Drop the duplicate rows from df dataframe

    Remove the existing categories column from the df dataframe and concat the formatted 
    categories dataframe with df dataframe.   

    Input: df
    Output: cleaned and formatted dataframe



    ''''


#Split the categories column in df dataframe and delimit the columns with ';'
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0].str.split('-', expand = True)
    categories.columns = list(row[0])

    #convert first row value in categories clolumns to labels
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        #set numerical values other than 0 and 1 as 1 by default
        for n, i in enumerate(categories[column]):
            if i > 1:
                categories[column][n] = 1


    #Drop Duplicates
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis=1, join="inner").drop_duplicates()

    

    return df
# In[4]:
def save_data(df, database_filename):

    '''
    Function to save the cleaned dataframe into a sql database with file name 'final'

    Input: df, database_filename
    Output: SQL Database 

    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('final', engine, index=False)

# In[5]:
def main():


    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# In[6]:
if __name__ == '__main__':
    main()
# %%
