import sys
import python as pd
import numpy as np
from sqlalchemy import create_engine
import re


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv('messages_filepath')
    categories = pd.read_csv('categories_filepath')
    df = messages.merge(categories, on = 'id', left_index=True, right_index=True)
    return df


def clean_data(df):

    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0].str.split('-', expand = True)
    categories.columns = list(row[0])

    #convert first row value in categories clolumns to labels
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
            #set
        for n, i in enumerate(categories[column]):
            if i > 1:
                categories[column][n] = 1

    #Drop Duplicates
    df.drop(['categories', 'original', 'genre', 'id'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis=1, join="inner").drop_duplicates()

    

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('database_filename', engine, index=False)


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


#if __name__ == '__main__':
    #main()