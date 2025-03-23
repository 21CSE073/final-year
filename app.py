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
    # Clean the data
    df['Text'] = df['Text'].str.strip()
    df['Sentiment'] = df['Sentiment'].str.strip()
    df['User'] = df['User'].str.strip()
    df['Hashtags'] = df['Hashtags'].str.strip()
    logger.info("Dataset loaded and cleaned successfully")

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
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for sentiment analysis")
            return {'sentiment': 'Neutral', 'confidence': 0.0, 'error': 'Invalid input text'}

        result = sentiment_analyzer(text)[0]
        label = result['label']
        score = result['score']

        # Map BERT labels to our sentiment categories
        if '1' in label or '2' in label:
            sentiment = 'Negative'
        elif '3' in label:
            sentiment = 'Neutral'
        else:
            sentiment = 'Positive'

        return {
            'sentiment': sentiment,
            'confidence': float(score),
            'raw_label': label,
            'error': None
        }
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        return {
            'sentiment': 'Neutral',
            'confidence': 0.0,
            'error': str(e)
        }


def create_user_profiles(df, topic):
    """Create user profiles from DataFrame using BERT model"""
    try:
        # Input validation
        if df is None or df.empty:
            logger.error("Input DataFrame is empty")
            return {'success': False, 'error': 'No data available', 'profiles': [], 'total_profiles': 0}

        if not topic or not isinstance(topic, str):
            logger.error("Invalid topic")
            return {'success': False, 'error': 'Invalid topic', 'profiles': [], 'total_profiles': 0}

        # Check required columns
        required_columns = ['User', 'Text', 'Hashtags']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {'success': False, 'error': f'Missing required columns: {missing_columns}', 'profiles': [],
                    'total_profiles': 0}

        # Filter posts for the topic
        filtered_df = df[
            df['Text'].str.contains(topic, case=False, na=False) |
            df['Hashtags'].str.contains(topic, case=False, na=False)
            ].copy()

        logger.info(f"Found {len(filtered_df)} posts for topic: {topic}")

        if filtered_df.empty:
            return {'success': False, 'error': f'No posts found for topic: {topic}', 'profiles': [],
                    'total_profiles': 0}

        # Analyze sentiments
        filtered_df['Sentiment'] = filtered_df['Text'].apply(lambda x: sentiment_analyzer(x[:512])[0]['label'])
        filtered_df['Confidence'] = filtered_df['Text'].apply(lambda x: sentiment_analyzer(x[:512])[0]['score'])

        # Group by user and calculate metrics
        user_profiles = []
        for user, group in filtered_df.groupby('User'):
            try:
                # Calculate sentiment counts
                sentiment_counts = group['Sentiment'].value_counts()
                positive_count = sentiment_counts.get('POSITIVE', 0)
                neutral_count = sentiment_counts.get('NEUTRAL', 0)
                negative_count = sentiment_counts.get('NEGATIVE', 0)

                # Get user's common hashtags
                hashtags = []
                if 'Hashtags' in group.columns:
                    hashtags = group['Hashtags'].dropna().str.split().explode().value_counts().head(5).index.tolist()

                # Get sample texts
                sample_texts = group['Text'].head(3).tolist()

                # Calculate average confidence
                avg_confidence = group['Confidence'].mean()

                profile = {
                    'User': user,
                    'Total_Posts': len(group),
                    'Positive_Posts': positive_count,
                    'Neutral_Posts': neutral_count,
                    'Negative_Posts': negative_count,
                    'Common_Hashtags': hashtags,
                    'Sample_Texts': sample_texts,
                    'Average_Confidence': round(avg_confidence, 2)
                }
                user_profiles.append(profile)
            except Exception as e:
                logger.error(f"Error processing user {user}: {str(e)}")
                continue

        if not user_profiles:
            return {'success': False, 'error': 'No valid user profiles created', 'profiles': [], 'total_profiles': 0}

        logger.info(f"Created {len(user_profiles)} user profiles")

        # Calculate sentiment distribution
        total_posts = len(filtered_df)
        if total_posts > 0:
            sentiment_distribution = {
                'positive': round((sum(p['Positive_Posts'] for p in user_profiles) / total_posts) * 100, 2),
                'neutral': round((sum(p['Neutral_Posts'] for p in user_profiles) / total_posts) * 100, 2),
                'negative': round((sum(p['Negative_Posts'] for p in user_profiles) / total_posts) * 100, 2)
            }
        else:
            sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}

        return {
            'success': True,
            'profiles': user_profiles,
            'total_profiles': len(user_profiles),
            'total_posts': total_posts,
            'sentiment_distribution': sentiment_distribution
        }

    except Exception as e:
        logger.error(f"Error creating user profiles: {str(e)}")
        return {'success': False, 'error': str(e), 'profiles': [], 'total_profiles': 0}


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
        # Get user's analyses and profiles
        analyses = data_manager.get_user_analyses(current_user.id)
        profiles = data_manager.get_user_profiles(current_user.id)

        # Process analyses for display
        processed_analyses = []
        for analysis in analyses:
            try:
                processed_analysis = {
                    'id': analysis.get('id'),
                    'title': analysis.get('title', ''),
                    'topic': analysis.get('topic', ''),
                    'created_at': analysis.get('timestamp', ''),
                    'total_posts': analysis.get('total_posts', 0),
                    'total_profiles': analysis.get('total_profiles', 0),
                    'sentiment_distribution': analysis.get('sentiment_distribution', {}),
                    'profiles': analysis.get('profiles', []),
                    'results': {
                        'topic': analysis.get('topic', ''),
                        'sentiment_distribution': analysis.get('sentiment_distribution', {}),
                        'profiles': analysis.get('profiles', [])
                    }
                }
                processed_analyses.append(processed_analysis)
            except Exception as e:
                logger.error(f"Error processing analysis: {str(e)}")
                continue

        # Calculate quick stats
        total_posts = sum(analysis.get('total_posts', 0) for analysis in analyses)
        total_profiles = sum(analysis.get('total_profiles', 0) for analysis in analyses)
        avg_confidence = sum(profile.get('average_confidence', 0) for profile in profiles) / len(
            profiles) if profiles else 0

        quick_stats = {
            'total_analyses': len(analyses),
            'total_profiles': total_profiles,
            'total_posts': total_posts,
            'average_confidence': round(avg_confidence, 2)
        }

        # Get recent topics
        recent_topics = [analysis.get('topic', '') for analysis in analyses[:5] if analysis.get('topic')]

        return render_template('dashboard.html',
                               analyses=processed_analyses,
                               profiles=profiles,
                               quick_stats=quick_stats,
                               recent_topics=recent_topics)

    except Exception as e:
        logger.error(f"Error in dashboard: {str(e)}")
        flash('An error occurred while loading the dashboard.')
        return redirect(url_for('home'))


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        analysis_type = data.get('type', 'topic')
        title = data.get('title', f'Analysis of {topic}')

        # Get posts for the topic
        posts = data_manager.get_posts_by_topic(topic)

        if not posts:
            return jsonify({
                'success': False,
                'error': 'No posts found for this topic'
            }), 404

        # Process posts and calculate sentiment
        processed_posts = []
        total_posts = len(posts)
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        user_sentiments = {}

        for post in posts:
            # Process text
            processed_text = process_text(post['text'])

            # Get sentiment prediction
            sentiment, confidence = predict_sentiment(processed_text)

            # Update sentiment counts
            if sentiment == 'positive':
                sentiment_counts['positive'] += 1
            elif sentiment == 'negative':
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

            # Update user sentiment counts
            username = post.get('username', 'Unknown')
            if username not in user_sentiments:
                user_sentiments[username] = {
                    'Positive_Posts': 0,
                    'Neutral_Posts': 0,
                    'Negative_Posts': 0,
                    'Total_Posts': 0,
                    'Average_Confidence': 0,
                    'User': username
                }

            user_sentiments[username]['Total_Posts'] += 1
            if sentiment == 'positive':
                user_sentiments[username]['Positive_Posts'] += 1
            elif sentiment == 'negative':
                user_sentiments[username]['Negative_Posts'] += 1
            else:
                user_sentiments[username]['Neutral_Posts'] += 1

            # Update average confidence
            current_avg = user_sentiments[username]['Average_Confidence']
            total_posts = user_sentiments[username]['Total_Posts']
            user_sentiments[username]['Average_Confidence'] = (current_avg * (
                        total_posts - 1) + confidence) / total_posts

            processed_posts.append({
                'text': post['text'],
                'sentiment': sentiment,
                'confidence': confidence,
                'username': username
            })

        # Calculate sentiment distribution percentages
        sentiment_distribution = {
            'positive': round((sentiment_counts['positive'] / total_posts) * 100, 1),
            'neutral': round((sentiment_counts['neutral'] / total_posts) * 100, 1),
            'negative': round((sentiment_counts['negative'] / total_posts) * 100, 1)
        }

        # Convert user_sentiments to list
        profiles = list(user_sentiments.values())

        # Save analysis
        analysis_id = data_manager.save_analysis(
            user_id=current_user.id,
            topic=topic,
            type=analysis_type,
            title=title,
            total_posts=total_posts,
            total_profiles=len(profiles),
            sentiment_distribution=sentiment_distribution,
            profiles=profiles,
            product_insights={},
            user_behavior={},
            recommendations=[]
        )

        return jsonify({
            'success': True,
            'data': {
                'id': analysis_id,
                'topic': topic,
                'type': analysis_type,
                'title': title,
                'total_posts': total_posts,
                'total_profiles': len(profiles),
                'sentiment_distribution': sentiment_distribution,
                'profiles': profiles,
                'summary': {
                    'total_posts': total_posts,
                    'total_profiles': len(profiles),
                    'sentiment_distribution': sentiment_distribution
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in analyze route: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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