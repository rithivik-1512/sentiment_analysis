import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("C:/Users/chsai/OneDrive/Desktop/new project/IMDB Dataset.csv")
df.sample(10)
print(df['sentiment'].value_counts())

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
df['cleaned_review'] = df['review'].apply(preprocess_text)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

x=df['cleaned_review']
y=df['sentiment_label']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# Combine xtest and ytest into a single DataFrame
test_df = pd.DataFrame({
    'review': xtest,
    'sentiment_label': ytest
})

# Optional: if you want the original sentiment strings instead of 0/1
# test_df['sentiment'] = test_df['sentiment_label'].map({1: 'positive', 0: 'negative'})

# Save to CSV
test_df['sentiment'] = test_df['sentiment_label'].map({1:'positive', 0:'negative'})
test_df = test_df.drop('sentiment_label',axis=1)
test_df.to_csv("test_df.csv", index=False)
print("Test dataset saved as test_df.csv")

# Save only the reviews in xtest
sample_df = pd.DataFrame({'review': xtest.values})  # use .values to get raw array
sample_df.to_csv("sample_df.csv", index=False)
print("Sample data for uploading saved")



xtrain_num = vectorizer.fit_transform(xtrain)
xtest_num = vectorizer.transform(xtest)

model=LogisticRegression(max_iter=5000,random_state=42)
model.fit(xtrain_num,ytrain)
ypred = model.predict(xtest_num)
print("accuracy score: ",accuracy_score(ytest,ypred))


joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

cm = confusion_matrix(ytest, ypred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

