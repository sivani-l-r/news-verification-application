from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import re

app = Flask(__name__)

# Load the pre-trained model and vectorizer
pac_model = joblib.load('pac_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


def fetch_content_from_link(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        return article_content
    except Exception as e:
        print(f"Error fetching article content: {e}")
        return ""


def clean_text(text):
    # Function to clean the text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text.strip()


@app.route('/')
def home():
    return render_template('index.html', prediction=None, article=None)


@app.route('/fetch', methods=['POST'])
def fetch():
    url = request.form['url']
    article_content = fetch_content_from_link(url)
    if not article_content:
        return render_template('index.html', prediction="Unable to fetch article content from URL.", article=None)
    else:
        # Vectorize the article content
        article_tfidf = tfidf_vectorizer.transform([clean_text(article_content)])
        # Predict using the model
        prediction = pac_model.predict(article_tfidf)[0]
        return render_template('index.html', prediction=prediction, article=article_content)


@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news-text']
    # Vectorize the news text
    news_tfidf = tfidf_vectorizer.transform([clean_text(news_text)])
    # Predict using the model
    prediction = pac_model.predict(news_tfidf)[0]
    return render_template('index.html', prediction=prediction, article=news_text)


if __name__ == '__main__':
    app.run(debug=True)
