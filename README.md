# Sentiment Analysis Project

This project predicts the sentiment of text reviews (**Positive** or **Negative**) using **Logistic Regression** and **TF-IDF vectorization**. It includes data preprocessing, model training, and a simple app for real-time predictions.

## Dataset

* **Test dataset:** `test_df.csv`  
* **Sample dataset:** `sample_df.csv`

You can use any labeled sentiment dataset (e.g., IMDB reviews, Amazon reviews, etc.).

## Features Used

* **Text reviews** (preprocessed for lowercase, punctuation removal, and stopwords)
* **TF-IDF features** extracted from the text

## Preprocessing Steps

1. Clean text:
   * Convert to lowercase
   * Remove punctuation
   * Remove stopwords (e.g., "the", "is", "and")
2. Transform text to numerical features using **TF-IDF Vectorization**

## Model

* **Algorithm:** Logistic Regression  
* **Library:** scikit-learn (`LogisticRegression`)

## How to Run

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn streamlit joblib matplotlib
```

2. (Optional) Retrain the model:

```bash
python NLP_coding.py
```

3. Run the app:
   * For CLI:

   ```bash
   python app_building.py
   ```

   * For Streamlit:

   ```bash
   streamlit run app_building.py
   ```

## Folder Structure

```
sentiment-analysis/
│
├── NLP_coding.py
├── app_building.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── test_df.csv
├── sample_df.csv
└── README.md
```

## Notes

* Model accuracy: ~89–92%
* Can be extended with additional preprocessing (lemmatization, n-grams) or classifiers (Random Forest, SVM, Neural Networks)
* Ideal for portfolio and demo purposes
