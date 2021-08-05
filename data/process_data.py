import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from the input files

    Args:
        categories_filepath (str): categories file's path
        messages_filepath (str): messages file's path

    Returns:
        df (pandas.DataFrame): dataframe containing the merged uncleaned dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    Clean the data

    Args:
        df (pandas.DataFrame): dataframe containing the merged uncleaned dataset

    Returns:
        df (pandas.DataFrame): dataframe containing the cleaned dataset
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    mask = np.where(df['related']==2,True, False)
    df.loc[df[mask].index, 'related'] = 1
    return df


def save_data(df, database_filepath):
    '''
    Save the data into the database. The destination table name is TABLE_NAME

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        database_filepath (str): database's path
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)


def main():
    '''
    Process the data and save it in a database

    Args:
        messages_filepath (str): messages file's path
        categories_filepath (str): categories file's path
        database_filepath (str): database file's path
    '''
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


if __name__ == '__main__':
    main()