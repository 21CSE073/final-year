import csv
import os
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging


class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.users_file = 'users.csv'
        self.analyses_file = 'analyses.csv'
        self.profiles_file = 'profiles.csv'
        self.sentiment_data = None
        self.flipkart_data = None
        self.combined_data = None
        self.reviews_data = None
        self._ensure_files_exist()
        self._load_datasets()

    def _load_datasets(self):
        """Load all datasets"""
        try:
            # Load sentiment dataset
            if os.path.exists('sentimentdataset.csv'):
                self.sentiment_data = pd.read_csv('sentimentdataset.csv')
                self.logger.info("Sentiment dataset loaded successfully")
            else:
                self.logger.warning("Sentiment dataset not found")

            # Load Flipkart dataset
            if os.path.exists('flipkart_product.csv'):
                self.flipkart_data = pd.read_csv('flipkart_product.csv')
                self.logger.info("Flipkart dataset loaded successfully")
            else:
                self.logger.warning("Flipkart dataset not found")

            # Load combined dataset
            if os.path.exists('combined.csv'):
                self.combined_data = pd.read_csv('combined.csv')
                self.logger.info("Combined dataset loaded successfully")
            else:
                self.logger.warning("Combined dataset not found")

            # Load reviews dataset
            if os.path.exists('reviews.csv'):
                self.reviews_data = pd.read_csv('reviews.csv')
                self.logger.info("Reviews dataset loaded successfully")
            else:
                self.logger.warning("Reviews dataset not found")

        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")

    def analyze_topic(self, topic):
        """Analyze a topic using sentiment dataset and BERT model"""
        try:
            # Initialize results
            results = {
                'topic': topic,
                'sentiment_distribution': {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                },
                'profiles': [],
                'total_posts': 0,
                'unique_users': 0,
                'average_likes': 0,
                'average_comments': 0
            }

            # Analyze sentiment data
            sentiment_results = self._analyze_sentiment_data(topic)
            if sentiment_results:
                results.update(sentiment_results)

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing topic: {str(e)}")
            return {
                'topic': topic,
                'sentiment_distribution': {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                },
                'profiles': [],
                'total_posts': 0,
                'unique_users': 0,
                'average_likes': 0,
                'average_comments': 0
            }

    def _analyze_sentiment_data(self, topic: str) -> Dict[str, Any]:
        """Analyze sentiment data for the given topic using sentimentdataset.csv"""
        if self.sentiment_data is None:
            self.logger.warning("Sentiment data is not loaded")
            return {}

        # Filter data for topic from sentiment dataset only
        topic_data = self.sentiment_data[
            self.sentiment_data['Text'].str.contains(topic, case=False, na=False)
        ]

        if topic_data.empty:
            self.logger.warning(f"No data found for topic: {topic}")
            return {}

        self.logger.info(f"Found {len(topic_data)} posts for topic: {topic}")

        # Calculate sentiment distribution using BERT model
        try:
            from transformers import pipeline
            sentiment_analyzer = pipeline("sentiment-analysis",
                                          model="nlptown/bert-base-multilingual-uncased-sentiment")

            # Analyze sentiment for each post
            sentiments = []
            for text in topic_data['Text']:
                try:
                    result = sentiment_analyzer(text[:512])[0]  # Limit text length for BERT
                    label = result['label']
                    if '1' in label or '2' in label:
                        sentiments.append('Negative')
                    elif '3' in label:
                        sentiments.append('Neutral')
                    else:
                        sentiments.append('Positive')
                except Exception as e:
                    self.logger.error(f"Error analyzing sentiment: {str(e)}")
                    sentiments.append('Neutral')

            topic_data['Sentiment'] = sentiments

            # Calculate sentiment distribution
            sentiment_counts = topic_data['Sentiment'].value_counts()
            total_posts = len(topic_data)
            sentiment_dist = {
                'positive': round((sentiment_counts.get('Positive', 0) / total_posts) * 100, 2),
                'neutral': round((sentiment_counts.get('Neutral', 0) / total_posts) * 100, 2),
                'negative': round((sentiment_counts.get('Negative', 0) / total_posts) * 100, 2)
            }

            # Create user profiles
            profiles = []
            unique_users = topic_data['User'].unique()
            self.logger.info(f"Found {len(unique_users)} unique users")

            for user in unique_users:
                user_data = topic_data[topic_data['User'] == user]

                # Calculate sentiment counts for this user
                positive_count = len(user_data[user_data['Sentiment'] == 'Positive'])
                neutral_count = len(user_data[user_data['Sentiment'] == 'Neutral'])
                negative_count = len(user_data[user_data['Sentiment'] == 'Negative'])

                # Get user's common hashtags
                hashtags = []
                if 'Hashtags' in user_data.columns:
                    hashtags = user_data['Hashtags'].dropna().str.split().explode().value_counts().head(
                        5).index.tolist()

                # Calculate engagement metrics
                avg_likes = user_data['Likes'].mean() if 'Likes' in user_data.columns else 0
                avg_comments = user_data['Comments'].mean() if 'Comments' in user_data.columns else 0

                profile = {
                    'User': user,
                    'Total_Posts': len(user_data),
                    'Positive_Posts': positive_count,
                    'Neutral_Posts': neutral_count,
                    'Negative_Posts': negative_count,
                    'Average_Likes': round(avg_likes, 2),
                    'Average_Comments': round(avg_comments, 2),
                    'Common_Hashtags': hashtags,
                    'Sentiment_Trend': {
                        'positive': round((positive_count / len(user_data)) * 100, 2),
                        'neutral': round((neutral_count / len(user_data)) * 100, 2),
                        'negative': round((negative_count / len(user_data)) * 100, 2)
                    }
                }
                profiles.append(profile)

            self.logger.info(f"Created {len(profiles)} user profiles")

            # Save profiles to CSV
            self._save_profiles_to_csv(profiles, topic)

            return {
                'sentiment_distribution': sentiment_dist,
                'profiles': profiles,
                'total_posts': len(topic_data),
                'unique_users': len(profiles),
                'average_likes': round(topic_data['Likes'].mean() if 'Likes' in topic_data.columns else 0, 2),
                'average_comments': round(topic_data['Comments'].mean() if 'Comments' in topic_data.columns else 0, 2)
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {}

    def _save_profiles_to_csv(self, profiles, topic):
        """Save profiles to CSV file"""
        try:
            # Read existing profiles
            existing_profiles = []
            if os.path.exists(self.profiles_file):
                existing_profiles = pd.read_csv(self.profiles_file).to_dict('records')

            # Add new profiles
            for profile in profiles:
                profile_data = {
                    'id': len(existing_profiles) + 1,
                    'user_id': 1,  # Default user ID
                    'topic': topic,
                    'total_posts': profile['Total_Posts'],
                    'positive_posts': profile['Positive_Posts'],
                    'neutral_posts': profile['Neutral_Posts'],
                    'negative_posts': profile['Negative_Posts'],
                    'common_hashtags': json.dumps(profile['Common_Hashtags']),
                    'sentiment_trend': json.dumps(profile['Sentiment_Trend']),
                    'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }
                existing_profiles.append(profile_data)

            # Save to CSV
            pd.DataFrame(existing_profiles).to_csv(self.profiles_file, index=False)
            self.logger.info(f"Saved {len(profiles)} profiles to CSV")

        except Exception as e:
            self.logger.error(f"Error saving profiles to CSV: {str(e)}")
            raise

    def _analyze_product_data(self, topic: str) -> Dict[str, Any]:
        """Analyze product data for the given topic"""
        if self.flipkart_data is None:
            return {}

        # Filter data for topic
        topic_data = self.flipkart_data[
            self.flipkart_data['Product Name'].str.contains(topic, case=False, na=False)
        ]

        if topic_data.empty:
            return {}

        # Calculate product insights
        insights = {
            'total_products': len(topic_data),
            'price_range': {
                'min': topic_data['Price'].min(),
                'max': topic_data['Price'].max(),
                'avg': topic_data['Price'].mean()
            },
            'rating_distribution': topic_data['Rating'].value_counts().to_dict(),
            'top_categories': topic_data['Category'].value_counts().head(5).to_dict()
        }

        return insights

    def _analyze_user_behavior(self, topic: str) -> Dict[str, Any]:
        """Analyze user behavior patterns for the given topic"""
        if self.combined_data is None:
            return {}

        # Filter data for topic
        topic_data = self.combined_data[
            self.combined_data['Text'].str.contains(topic, case=False, na=False)
        ]

        if topic_data.empty:
            return {}

        # Analyze user behavior
        behavior = {
            'posting_patterns': self._analyze_posting_patterns(topic_data),
            'content_preferences': self._analyze_content_preferences(topic_data),
            'engagement_patterns': self._analyze_engagement_patterns(topic_data)
        }

        return behavior

    def _generate_recommendations(self, topic: str) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'content_strategy': self._generate_content_strategy(topic),
            'engagement_tactics': self._generate_engagement_tactics(topic),
            'timing_optimization': self._generate_timing_recommendations(topic),
            'hashtag_strategy': self._generate_hashtag_strategy(topic)
        }
        return recommendations

    def _analyze_posting_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when users post about the topic"""
        if 'Timestamp' not in data.columns:
            return {}

        data['hour'] = pd.to_datetime(data['Timestamp']).dt.hour
        hourly_dist = data['hour'].value_counts().sort_index().to_dict()

        return {
            'hourly_distribution': hourly_dist,
            'peak_hours': sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def _analyze_content_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze what type of content performs best"""
        if 'Content Type' not in data.columns:
            return {}

        content_performance = data.groupby('Content Type')['Engagement'].mean().to_dict()
        return {
            'content_performance': content_performance,
            'best_performing_type': max(content_performance.items(), key=lambda x: x[1])[0]
        }

    def _analyze_engagement_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how users engage with content"""
        if 'Engagement' not in data.columns:
            return {}

        return {
            'avg_engagement': data['Engagement'].mean(),
            'engagement_trend': self._calculate_engagement_trend(data)
        }

    def _calculate_engagement_trend(self, data: pd.DataFrame) -> str:
        """Calculate the trend in engagement over time"""
        if 'Timestamp' not in data.columns or 'Engagement' not in data.columns:
            return "Insufficient data"

        data['date'] = pd.to_datetime(data['Timestamp']).dt.date
        daily_engagement = data.groupby('date')['Engagement'].mean()

        if len(daily_engagement) < 2:
            return "Insufficient data"

        trend = daily_engagement.iloc[-1] - daily_engagement.iloc[0]
        if trend > 0:
            return "Increasing"
        elif trend < 0:
            return "Decreasing"
        else:
            return "Stable"

    def _generate_content_strategy(self, topic: str) -> List[str]:
        """Generate content strategy recommendations"""
        recommendations = []

        # Analyze sentiment data
        sentiment_insights = self._analyze_sentiment_data(topic)
        if sentiment_insights:
            sentiment_dist = sentiment_insights.get('sentiment_distribution', {})
            if sentiment_dist.get('Positive', 0) > 0.6:
                recommendations.append("Focus on positive and uplifting content")
            elif sentiment_dist.get('Negative', 0) > 0.6:
                recommendations.append("Address concerns and provide solutions")

        # Analyze product data
        product_insights = self._analyze_product_data(topic)
        if product_insights:
            price_range = product_insights.get('price_range', {})
            if price_range:
                recommendations.append(f"Highlight products in the {price_range['avg']:.2f} price range")

        return recommendations

    def _generate_engagement_tactics(self, topic: str) -> List[str]:
        """Generate engagement tactics recommendations"""
        tactics = []

        # Analyze user behavior
        behavior = self._analyze_user_behavior(topic)
        posting_patterns = behavior.get('posting_patterns', {})

        if posting_patterns.get('peak_hours'):
            peak_hour = posting_patterns['peak_hours'][0][0]
            tactics.append(f"Post during peak engagement hours (around {peak_hour}:00)")

        content_preferences = behavior.get('content_preferences', {})
        if content_preferences.get('best_performing_type'):
            tactics.append(f"Focus on {content_preferences['best_performing_type']} content")

        return tactics

    def _generate_timing_recommendations(self, topic: str) -> List[str]:
        """Generate timing optimization recommendations"""
        recommendations = []

        behavior = self._analyze_user_behavior(topic)
        posting_patterns = behavior.get('posting_patterns', {})

        if posting_patterns.get('hourly_distribution'):
            hourly_dist = posting_patterns['hourly_distribution']
            best_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(f"Best posting times: {', '.join(f'{hour}:00' for hour, _ in best_hours)}")

        return recommendations

    def _generate_hashtag_strategy(self, topic: str) -> List[str]:
        """Generate hashtag strategy recommendations"""
        recommendations = []

        sentiment_insights = self._analyze_sentiment_data(topic)
        if sentiment_insights.get('top_hashtags'):
            top_hashtags = list(sentiment_insights['top_hashtags'].keys())[:5]
            recommendations.append(f"Use popular hashtags: {', '.join(top_hashtags)}")

        return recommendations

    def _ensure_files_exist(self):
        """Ensure all required CSV files exist with headers."""
        files = {
            self.users_file: ['id', 'username', 'email', 'password_hash', 'created_at', 'is_admin'],
            self.analyses_file: ['id', 'user_id', 'title', 'analysis_type', 'content', 'created_at', 'results'],
            self.profiles_file: ['id', 'user_id', 'topic', 'total_posts', 'positive_posts', 'neutral_posts',
                                 'negative_posts', 'common_hashtags', 'sentiment_trend', 'created_at', 'updated_at']
        }

        for file_path, headers in files.items():
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()

    def _get_next_id(self, file_path):
        """Get the next available ID for a CSV file."""
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                ids = [int(row['id']) for row in reader]
                return max(ids) + 1 if ids else 1
        except FileNotFoundError:
            return 1

    def get_user_by_id(self, user_id):
        """Get user by ID."""
        with open(self.users_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['id']) == user_id:
                    return row
        return None

    def create_user(self, username, email, password, is_admin=False):
        """Create a new user."""
        if self.get_user_by_username(username):
            return None

        user_id = self._get_next_id(self.users_file)
        user = {
            'id': user_id,
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'is_admin': str(is_admin)  # Convert boolean to string
        }

        with open(self.users_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=user.keys())
            writer.writerow(user)

        return user

    def get_user_by_username(self, username):
        """Get user by username."""
        with open(self.users_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username:
                    # Ensure is_admin field exists
                    if 'is_admin' not in row:
                        row['is_admin'] = 'False'
                    return row
        return None

    def verify_password(self, username, password):
        """Verify user password."""
        user = self.get_user_by_username(username)
        if user:
            return check_password_hash(user['password_hash'], password)
        return False

    def save_analysis(self, user_id, analysis_data):
        """Save analysis results"""
        try:
            analyses = []
            if os.path.exists(self.analyses_file):
                analyses = pd.read_csv(self.analyses_file).to_dict('records')

            analysis_data['user_id'] = user_id
            analysis_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            analyses.append(analysis_data)

            pd.DataFrame(analyses).to_csv(self.analyses_file, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")
            return False

    def get_user_analyses(self, user_id):
        """Get all analyses for a user."""
        analyses = []
        try:
            if os.path.exists(self.analyses_file):
                analyses_df = pd.read_csv(self.analyses_file)
                user_analyses = analyses_df[analyses_df['user_id'] == user_id].to_dict('records')

                # Convert timestamp to created_at for compatibility
                for analysis in user_analyses:
                    if 'timestamp' in analysis:
                        analysis['created_at'] = analysis['timestamp']
                        del analysis['timestamp']
                    if 'results' in analysis:
                        try:
                            analysis['results'] = json.loads(analysis['results'])
                        except (json.JSONDecodeError, KeyError):
                            analysis['results'] = {}
                    analyses.append(analysis)

            return sorted(analyses, key=lambda x: x.get('created_at', ''), reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting user analyses: {str(e)}")
            return []

    def save_user_profile(self, user_id, topic, profile_data):
        """Save a new user profile."""
        profile_id = self._get_next_id(self.profiles_file)
        profile = {
            'id': profile_id,
            'user_id': user_id,
            'topic': topic,
            'total_posts': profile_data['Total Posts'],
            'positive_posts': profile_data['Positive Posts'],
            'neutral_posts': profile_data['Neutral Posts'],
            'negative_posts': profile_data['Negative Posts'],
            'common_hashtags': json.dumps(profile_data['Common Hashtags']),
            'sentiment_trend': json.dumps(profile_data['Sentiment Trend']),
            'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.profiles_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=profile.keys())
            writer.writerow(profile)

        return profile

    def get_user_profiles(self, user_id):
        """Get all profiles associated with a user."""
        try:
            if not os.path.exists('profiles.csv'):
                return []

            profiles_df = pd.read_csv('profiles.csv')
            user_profiles = profiles_df[profiles_df['user_id'] == user_id].to_dict('records')

            # Convert string representations back to lists/dicts
            for profile in user_profiles:
                if 'hashtags' in profile:
                    profile['hashtags'] = eval(profile['hashtags'])
                if 'sentiment_trend' in profile:
                    profile['sentiment_trend'] = eval(profile['sentiment_trend'])

            return user_profiles
        except Exception as e:
            logging.error(f"Error getting user profiles: {str(e)}")
            return []

    def get_quick_stats(self, user_id):
        """Get quick statistics for the dashboard."""
        analyses = self.get_user_analyses(user_id)
        profiles = self.get_user_profiles(user_id)

        return {
            'total_analyses': len(analyses),
            'total_profiles': len(profiles)
        } 