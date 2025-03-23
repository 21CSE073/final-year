from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from data_manager import DataManager
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize extensions
data_manager = DataManager()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

try:
    # Load the dataset
    df = pd.read_csv('sentimentdataset.csv')
    logger.info("Dataset loaded successfully")

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    logger.info("BERT model loaded successfully")
except Exception as e:
    logger.error(f"Error loading data or model: {str(e)}")
    sentiment_analyzer = None


class User(UserMixin):
    def __init__(self, user_data):
        self.id = int(user_data['id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.is_admin = user_data.get('is_admin', 'False') == 'True'


@login_manager.user_loader
def load_user(user_id):
    user_data = data_manager.get_user_by_id(int(user_id))
    if user_data:
        return User(user_data)
    return None


# Preprocess the data
def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        return ""


# Prepare the data
X = df['Text'].apply(preprocess_text)
y = df['Sentiment'].str.strip()

# Create and fit the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_vectorized, y)


def predict_sentiment(text):
    try:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        if '1' in label or '2' in label:
            return 'Negative'
        elif '3' in label:
            return 'Neutral'
        else:
            return 'Positive'
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        return 'Neutral'


def create_user_profiles(df, topic):
    try:
        logger.info(f"Creating profiles for topic: {topic}")

        # Check if required columns exist
        required_columns = ['Text', 'Hashtags', 'User', 'Timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return []

        # Create a copy of the filtered dataframe to avoid the SettingWithCopyWarning
        topic_filtered_df = df[
            df['Text'].str.contains(topic, case=False, na=False) |
            df['Hashtags'].str.contains(topic, case=False, na=False)
            ].copy()

        if topic_filtered_df.empty:
            logger.warning(f"No posts found for topic: {topic}")
            return []

        logger.info(f"Found {len(topic_filtered_df)} posts for topic: {topic}")

        # Analyze sentiments using BERT
        try:
            topic_filtered_df.loc[:, 'Predicted_Sentiment'] = topic_filtered_df['Text'].apply(predict_sentiment)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return []

        # Group by User and analyze sentiments
        user_profiles = []
        grouped_users = topic_filtered_df.groupby('User')

        for user, group in grouped_users:
            try:
                # Create a copy of the group to avoid SettingWithCopyWarning
                group_copy = group.copy()

                # Calculate sentiment trends over time
                try:
                    group_copy['Timestamp'] = pd.to_datetime(group_copy['Timestamp'])
                    sentiment_trend = group_copy.groupby(group_copy['Timestamp'].dt.date)[
                        'Predicted_Sentiment'].value_counts(normalize=True).unstack(fill_value=0)

                    # Convert the sentiment trend DataFrame to a dictionary with string dates
                    sentiment_trend_dict = {}
                    for date in sentiment_trend.index:
                        date_str = date.strftime('%Y-%m-%d')
                        sentiment_trend_dict[date_str] = sentiment_trend.loc[date].to_dict()
                except Exception as e:
                    logger.error(f"Error processing sentiment trend for user {user}: {str(e)}")
                    sentiment_trend_dict = {}

                # Get sentiment counts
                sentiment_counts = group_copy['Predicted_Sentiment'].value_counts()

                # Get hashtags, handling NaN values
                hashtags = []
                if 'Hashtags' in group_copy.columns:
                    hashtags = [str(tag) for tag in group_copy['Hashtags'].unique().tolist() if pd.notna(tag)]

                profile = {
                    'User Name': str(user),
                    'Total Posts': int(len(group_copy)),
                    'Positive Posts': int(sentiment_counts.get('Positive', 0)),
                    'Neutral Posts': int(sentiment_counts.get('Neutral', 0)),
                    'Negative Posts': int(sentiment_counts.get('Negative', 0)),
                    'Sample Texts': group_copy['Text'].head(3).tolist(),
                    'Common Hashtags': hashtags,
                    'Sentiment Trend': sentiment_trend_dict
                }
                user_profiles.append(profile)
            except Exception as e:
                logger.error(f"Error processing user {user}: {str(e)}")
                continue

        logger.info(f"Created {len(user_profiles)} user profiles")
        return user_profiles
    except Exception as e:
        logger.error(f"Error in create_user_profiles: {str(e)}")
        return []


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if data_manager.verify_password(username, password):
            user_data = data_manager.get_user_by_username(username)
            user = User(user_data)
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        user = data_manager.create_user(username, email, password)
        if user:
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        flash('Username or email already exists')
    return render_template('signup.html')


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get user's analyses
        analyses = data_manager.get_user_analyses(current_user.id)

        # Get user's profiles
        user_profiles = data_manager.get_user_profiles(current_user.id)

        # Get quick stats
        stats = data_manager.get_quick_stats(current_user.id)

        # Log the data for debugging
        app.logger.info(f"Found {len(analyses)} analyses and {len(user_profiles)} profiles for user {current_user.id}")

        return render_template('dashboard.html',
                               analyses=analyses,
                               user_profiles=user_profiles,
                               stats=stats)
    except Exception as e:
        app.logger.error(f"Error in dashboard: {str(e)}")
        flash('An error occurred while loading the dashboard. Please try again.', 'error')
        return redirect(url_for('home'))


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        data = request.get_json()
        analysis_type = data.get('type')
        title = data.get('title', 'Untitled Analysis')

        if analysis_type == 'text':
            text = data.get('text', '')
            if not text:
                return jsonify({'error': 'No text provided'}), 400

            # Perform sentiment analysis
            sentiment = analyze_sentiment(text)
            result = {
                'type': 'text',
                'title': title,
                'content': text,
                'sentiment': sentiment['label'],
                'confidence': round(sentiment['score'] * 100, 2),
                'sentiment_color': get_sentiment_color(sentiment['label']),
                'sentiment_icon': get_sentiment_icon(sentiment['label'])
            }

        elif analysis_type == 'topic':
            topic = data.get('topic', '')
            if not topic:
                return jsonify({'error': 'No topic provided'}), 400

            # Perform topic analysis
            analysis_results = data_manager.analyze_topic(topic)

            # Ensure all required fields are present
            result = {
                'type': 'topic',
                'title': title,
                'topic': topic,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment_distribution': analysis_results.get('sentiment_distribution', {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                }),
                'profiles': analysis_results.get('profiles', []),
                'total_posts': analysis_results.get('total_posts', 0),
                'unique_users': analysis_results.get('unique_users', 0),
                'average_likes': analysis_results.get('average_likes', 0),
                'average_comments': analysis_results.get('average_comments', 0)
            }

            # Save user profiles
            if result['profiles']:
                for profile in result['profiles']:
                    data_manager.save_user_profile(
                        current_user.id,
                        topic,
                        {
                            'Total Posts': profile['Total_Posts'],
                            'Positive Posts': profile['Positive_Posts'],
                            'Neutral Posts': profile['Neutral_Posts'],
                            'Negative Posts': profile['Negative_Posts'],
                            'Common Hashtags': profile['Common_Hashtags'],
                            'Sentiment Trend': profile['Sentiment_Trend']
                        }
                    )

        else:
            return jsonify({'error': 'Invalid analysis type'}), 400

        # Save analysis
        if not data_manager.save_analysis(current_user.id, result):
            return jsonify({'error': 'Failed to save analysis'}), 500

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in analyze route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


def get_sentiment_color(sentiment):
    """Get color code for sentiment visualization"""
    colors = {
        'Positive': '#28a745',  # Green
        'Negative': '#dc3545',  # Red
        'Neutral': '#ffc107'  # Yellow
    }
    return colors.get(sentiment, '#6c757d')  # Default gray


def get_sentiment_icon(sentiment):
    """Get emoji icon for sentiment visualization"""
    icons = {
        'Positive': '😊',
        'Negative': '😔',
        'Neutral': '😐'
    }
    return icons.get(sentiment, '❓')


def analyze_sentiment(text):
    """Analyze sentiment of text using BERT model"""
    try:
        if sentiment_analyzer is None:
            logger.error("BERT model not initialized")
            return {'label': 'Neutral', 'score': 0.0}

        # Get model prediction
        result = sentiment_analyzer(text[:512])[0]  # Limit text length for BERT
        label = result['label']

        # Convert BERT labels to our format
        if '1' in label or '2' in label:
            return {'label': 'Negative', 'score': result['score']}
        elif '3' in label:
            return {'label': 'Neutral', 'score': result['score']}
        else:
            return {'label': 'Positive', 'score': result['score']}

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {'label': 'Neutral', 'score': 0.0}


if __name__ == '__main__':
    app.run(debug=True) 