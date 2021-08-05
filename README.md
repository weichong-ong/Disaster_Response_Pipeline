# Overview
This project is part of [Data Science Nanodegree Program by Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) in collaboration with [Figure Eight](https://appen.com). The dataset contains real messages that were sent during disaster events. The purpose of this project is to build machine learning pipeline, which consists of Natural Language Processing model and Classification model, to categorize these events so that these messages can be sent to an appropriate disaster relief agency. The project will include a web app where an emergency worker can input a new message and get classification results in several categories.

# Components
The project is divided into three components:

1. **ETL Pipeline:** Load datasets, clean the data and store it in a SQLite database
2. **ML Pipeline:** Build a text processing and machine learning pipeline, train a model to classify text message in categories
3. **Flask Web App:** Show model results in real time

# Requirements
- Python 3.8
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy

# Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


# Classification Model
The dataset is imbalanced (ie some labels like water have few examples) which contributes to lower score for the ones which have less data. That is why the overall score is low. For skewed datasets, accuracy as a metric for model evaluation is not preferred. In this case, it's better to have more FP than FN, as it makes sure than the messages are at least sent to appropriate disaster relief agency, instead of ignoring some really urgent messages (FN). Therefore, it is important to have a classification model that shows low number of FN -> high recall value. 

<p align="center">
  <img src="/images/classification_report.png" height="600" width="800" />
</p>

# Acknowledgements
- [Udacity](https://www.udacity.com) for providing such a interesting and meaningful project
- [Figure Eight](https://appen.com) for providing real-world dataset 

# Results
1. Input message to get classification results
<p align="center">
  <img src="/images/message_input.png" height="200" width="800" />
</p>

2. Example: The categories which the message belongs to highlighted in green
<p align="center">
  <img src="/images/result_example.png" height="600" width="800" />
</p>

3. Overview of Training Dataset
<p align="center">
  <img src="/images/distribution_genre.png" height="400" width="800" />
</p>

<p align="center">
  <img src="/images/distribution_catogory.png" height="400" width="800" />
</p>
