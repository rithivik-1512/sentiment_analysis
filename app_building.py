import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import precision_score, recall_score


# ---------------------------
# configuration 
# ---------------------------

st.set_page_config(
    page_title='Sentiment analysis dashboard',
    page_icon='🎭',
    layout='centered'
)

# ---------------------------
# preprocessing
# ---------------------------
def preprocess_text(text):
    text=text.lower()
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# training the model
# ---------------------------

def train_model(df):
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    test_x = df['review']
    test_y = df['sentiment_label']

    test_num = vectorizer.transform(test_x)
    ypred = model.predict(test_num)

    accuracy = accuracy_score(test_y, ypred)
    precision = precision_score(test_y, ypred)
    recall = recall_score(test_y, ypred)

    return{
        "model":model,
        "vectorizer":vectorizer,
        "test_y":test_y,
        "ypred":ypred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

# ---------------------------
# main app
# ---------------------------

def main():
    st.header("🎭 Sentiment analysis dashboard")
    
    st.sidebar.title("navigation")
    page = st.sidebar.radio(
        "select page",
        ["home page","upload and predict","post and predict"]
    )
    
    if page == "home page":
        
        df = pd.read_csv("test_df.csv")
        st.write("preview of the uploaded data")
        st.write(df.sample(10))
        
        st.write("-------")
        st.write("This interactive dashboard shows the performace of the trained model for the IMDB reviews dataset")

        results = train_model(df)
        test_y = results["test_y"]
        ypred = results["ypred"]
        accuracy = results["accuracy"]
        precision = results["precision"]
        recall = results["recall"]

        col1, col2, col3 = st.columns(3)
        with col1:
          st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
          st.metric("Precision", f"{precision:.3f}")
        with col3:
          st.metric("Recall", f"{recall:.3f}")
        
        cm = confusion_matrix(test_y, ypred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    
    if page == "upload and predict":
      st.subheader("Instructions for uploading CSV file:")
      st.markdown("""
       - The CSV file should contain **one column**: `review`.
       - The `review` column should have **text data** (movie/product reviews, etc.).
       - **No null/empty values** should be present in the column.
       - File type must be **CSV** (`.csv`).
      """)

      uploaded_file = st.file_uploader("Select or browse a CSV file", type=['csv'])
    
      if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        elif df['review'].isnull().any():
            st.error("Please remove null/empty values from the 'review' column.")
        else:
            st.success("File uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.write(df.sample(5))

            # Load trained model and vectorizer
            model = joblib.load("sentiment_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")

            # Preprocess the uploaded reviews
            df['cleaned_review'] = df['review'].apply(preprocess_text)
            
            # Transform using the vectorizer
            xtest_num = vectorizer.transform(df['cleaned_review'])

            # Predict sentiment
            ypred = model.predict(xtest_num)
            df['sentiment'] = ypred
            df['sentiment'] = df['sentiment'].map({1: "positive", 0: "negative"})

            st.write("Prediction results:")
            st.write(df[['review', 'sentiment']].sample(10))
    if page == "post and predict":
      st.subheader("Enter a review to predict sentiment:")
      user_input = st.text_area("Type your review here:")

      if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a review to predict sentiment.")
        else:
            # Preprocess the text
            cleaned_text = preprocess_text(user_input)

            # Load model and vectorizer
            model = joblib.load("sentiment_model.pkl")
            vectorizer = joblib.load("tfidf_vectorizer.pkl")

            # Transform input
            text_num = vectorizer.transform([cleaned_text])

            # Predict
            prediction = model.predict(text_num)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"

            st.success(f"The predicted sentiment is: **{sentiment}**")


        

if __name__ == "__main__":
    main()