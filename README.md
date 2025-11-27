Sentiment Analysis Project

A machine learning based sentiment classifier built using Logistic Regression and TF-IDF vectorization. This repository includes the full training pipeline, model artifacts, test datasets, and a deployment-ready application script.

Project Structure
.
├── NLP_coding.py              # Training, preprocessing, evaluation, model saving
├── app_building.py            # UI app for real-time sentiment prediction
├── sentiment_model.pkl        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── test_df.csv                # Test dataset
├── sample_df.csv              # Sample dataset
└── README.md                  # Documentation

Overview

This project performs binary sentiment classification (Positive/Negative) on text reviews using:

TF-IDF Vectorization

Logistic Regression

Scikit-Learn, NumPy, Pandas

Streamlit/CLI application for predictions

It is suitable for quick demonstrations, academic projects, and portfolio use.

How It Works

Load IMDb-style text dataset

Clean and preprocess the text

Convert text to numerical features using TF-IDF

Train Logistic Regression model

Save the trained model and vectorizer

Use the app to classify user-entered text as Positive or Negative

Files Description
NLP_coding.py

Loads dataset

Preprocesses text

Trains Logistic Regression model

Saves sentiment_model.pkl and tfidf_vectorizer.pkl

Evaluates the model

app_building.py

Loads the trained model and vectorizer

Accepts user input

Performs TF-IDF transformation

Predicts sentiment in real time

sentiment_model.pkl

Serialized trained ML model.

tfidf_vectorizer.pkl

Serialized TF-IDF vectorizer fitted on training data.

test_df.csv / sample_df.csv

Datasets used for testing and demonstration.

How to Run
1. Install Dependencies
pip install numpy pandas scikit-learn streamlit joblib matplotlib

2. (Optional) Retrain the Model
python NLP_coding.py

3. Run the Application

If it is a Python CLI app:

python app_building.py


If it is a Streamlit app:

streamlit run app_building.py

Model Performance

Accuracy: ~89–92% (varies depending on dataset split)

Balanced precision and recall

Technologies Used

Python 3.x

Scikit-Learn

Pandas

NumPy

Streamlit

Joblib
