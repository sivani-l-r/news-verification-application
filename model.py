import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_and_save_model():
    # Load dataset
    data = pd.read_csv('news.csv')

    # Train-test split
    labels = data.label
    x_train, x_test, y_train, y_test = train_test_split(data['text'], labels, test_size=0.2, random_state=7)

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    y_pred = pac.predict(tfidf_test)

    # Model evaluation
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')

    # Save the model and vectorizer
    joblib.dump(pac, 'pac_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


def load_model_and_vectorizer():
    # Load the pre-trained model and vectorizer
    pac_model = joblib.load('pac_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return pac_model, tfidf_vectorizer
