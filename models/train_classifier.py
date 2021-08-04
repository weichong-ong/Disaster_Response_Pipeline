import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'],axis=1).values
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__learning_rate': [0.1, 0.5]
    }
    # create grid search object
    cv = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 3, n_jobs=-1, verbose=2)

    return cv

def evaluate_model(y_test, y_pred, category):
    clsReport = classification_report(y_test, y_pred, target_names = category, zero_division=1)
    print("Classification report:", clsReport)

def save_model(model, classifier_filepath):
    pickle.dump(model, open(classifier_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:

        database_filepath, classifier_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        print('Training model...\n')    
        model = build_model()      
        model.fit(X_train, y_train)

        print('Predicting on test data...\n') 
        y_pred = model.predict(X_test)

        print('Evaluating model...\n')
        evaluate_model(y_test, y_pred, category)
        
        print('Saving model...\n    MODEL: {}'.format(classifier_filepath))
        save_model(model, classifier_filepath)
        
        print('Trained model saved!')
    
    else:
        print('Please provide the filepath of the disaster response database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')

if __name__ == '__main__':
    main()