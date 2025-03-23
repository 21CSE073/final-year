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
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

            # Load sentiment dataset
            if os.path.exists('sentimentdataset.csv'):
                for encoding in encodings:
                    try:
                        self.sentiment_data = pd.read_csv('sentimentdataset.csv', encoding=encoding)
                        self.logger.info(f"Sentiment dataset loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error loading sentiment dataset with {encoding} encoding: {str(e)}")
                        continue
                else:
                    self.logger.warning("Could not load sentiment dataset with any supported encoding")
            else:
                self.logger.warning("Sentiment dataset not found")

            # Load Flipkart dataset
            if os.path.exists('flipkart_product.csv'):
                for encoding in encodings:
                    try:
                        self.flipkart_data = pd.read_csv('flipkart_product.csv', encoding=encoding)
                        self.logger.info(f"Flipkart dataset loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error loading Flipkart dataset with {encoding} encoding: {str(e)}")
                        continue
                else:
                    self.logger.warning("Could not load Flipkart dataset with any supported encoding")
            else:
                self.logger.warning("Flipkart dataset not found")

            # Load combined dataset
            if os.path.exists('combined.csv'):
                for encoding in encodings:
                    try:
                        self.combined_data = pd.read_csv('combined.csv', encoding=encoding)
                        self.logger.info(f"Combined dataset loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error loading combined dataset with {encoding} encoding: {str(e)}")
                        continue
                else:
                    self.logger.warning("Could not load combined dataset with any supported encoding")
            else:
                self.logger.warning("Combined dataset not found")

            # Load reviews dataset
            if os.path.exists('reviews.csv'):
                for encoding in encodings:
                    try:
                        self.reviews_data = pd.read_csv('reviews.csv', encoding=encoding)
                        self.logger.info(f"Reviews dataset loaded successfully with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        self.logger.error(f"Error loading reviews dataset with {encoding} encoding: {str(e)}")
                        continue
                else:
                    self.logger.warning("Could not load reviews dataset with any supported encoding")
            else:
                self.logger.warning("Reviews dataset not found")

        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")

    def analyze_topic(self, topic):
        """Analyze a topic using sentiment dataset and BERT model"""
        try:
            # Initialize results with all required fields
            results = {
                'topic': topic,
                'type': 'sentiment',
                'title': f"Analysis for {topic}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment_distribution': {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                },
                'profiles': [],
                'total_posts': 0,
                'unique_users': 0,
                'average_likes': 0,
                'average_comments': 0,
                'total_profiles': 0,
                'product_insights': {},
                'user_behavior': {},
                'recommendations': {}
            }

            # Analyze sentiment data
            sentiment_results = self._analyze_sentiment_data(topic)
            if sentiment_results:
                # Update only the fields that exist in sentiment_results
                for key in results.keys():
                    if key in sentiment_results:
                        results[key] = sentiment_results[key]

            return results

        except Exception as e:
            self.logger.error(f"Error analyzing topic: {str(e)}")
            # Return a properly structured empty result
            return {
                'topic': topic,
                'type': 'sentiment',
                'title': f"Analysis for {topic}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sentiment_distribution': {
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0
                },
                'profiles': [],
                'total_posts': 0,
                'unique_users': 0,
                'average_likes': 0,
                'average_comments': 0,
                'total_profiles': 0,
                'product_insights': {},
                'user_behavior': {},
                'recommendations': {}
            }

    def _analyze_sentiment_data(self, topic: str) -> Dict[str, Any]:
        """Analyze sentiment data for the given topic using sentimentdataset.csv"""
        if self.sentiment_data is None:
            self.logger.warning("Sentiment data is not loaded")
            return {}

        try:
            # Clean and prepare the data
            self.sentiment_data['Text'] = self.sentiment_data['Text'].str.strip()
            self.sentiment_data['Sentiment'] = self.sentiment_data['Sentiment'].str.strip()
            self.sentiment_data['User'] = self.sentiment_data['User'].str.strip()
            self.sentiment_data['Hashtags'] = self.sentiment_data['Hashtags'].str.strip()

            # Filter data for topic
            topic_data = self.sentiment_data[
                self.sentiment_data['Text'].str.contains(topic, case=False, na=False)
            ]

            if topic_data.empty:
                self.logger.warning(f"No data found for topic: {topic}")
                return {}

            self.logger.info(f"Found {len(topic_data)} posts for topic: {topic}")

            # Calculate sentiment distribution
            sentiment_counts = topic_data['Sentiment'].value_counts()
            total_posts = len(topic_data)
            sentiment_dist = {
                'positive': round((sentiment_counts.get('Positive', 0) / total_posts) * 100, 2),
                'neutral': round((sentiment_counts.get('Neutral', 0) / total_posts) * 100, 2),
                'negative': round((sentiment_counts.get('Negative', 0) / total_posts) * 100, 2)
            }

            # Calculate engagement metrics
            avg_likes = topic_data['Likes'].mean() if 'Likes' in topic_data.columns else 0
            avg_comments = topic_data['Retweets'].mean() if 'Retweets' in topic_data.columns else 0

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

                # Get sample texts
                sample_texts = user_data['Text'].head(3).tolist()

                profile = {
                    'User': user,
                    'Total_Posts': len(user_data),
                    'Positive_Posts': positive_count,
                    'Neutral_Posts': neutral_count,
                    'Negative_Posts': negative_count,
                    'Common_Hashtags': hashtags,
                    'Sample_Texts': sample_texts,
                    'Average_Confidence': 0.8  # Default confidence value
                }
                profiles.append(profile)

            self.logger.info(f"Created {len(profiles)} user profiles")

            return {
                'sentiment_distribution': sentiment_dist,
                'profiles': profiles,
                'total_posts': len(topic_data),
                'unique_users': len(unique_users),
                'average_likes': round(avg_likes, 2),
                'average_comments': round(avg_comments, 2),
                'total_profiles': len(profiles),
                'product_insights': {},
                'user_behavior': {},
                'recommendations': {}
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
            'analyses.csv': ['id', 'user_id', 'title', 'analysis_type', 'content', 'created_at', 'results'],
            'user_profiles.csv': ['id', 'user_id', 'username', 'total_posts', 'sentiment_counts',
                                  'common_hashtags', 'sample_texts', 'average_confidence', 'created_at']
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

    def save_analysis(self, user_id: int, analysis_data: Dict[str, Any]) -> bool:
        """Save analysis results to analyses.csv"""
        try:
            # Read existing analyses
            analyses_file = 'analyses.csv'
            existing_analyses = []
            if os.path.exists(analyses_file):
                with open(analyses_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_analyses = list(reader)

            # Create new analysis record
            new_analysis = {
                'id': len(existing_analyses) + 1,
                'user_id': user_id,
                'topic': analysis_data.get('topic', ''),
                'type': 'topic',
                'title': analysis_data.get('title', ''),
                'timestamp': analysis_data.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'total_posts': analysis_data.get('total_posts', 0),
                'total_profiles': analysis_data.get('total_profiles', 0),
                'sentiment_distribution': json.dumps(analysis_data.get('sentiment_distribution', {})),
                'profiles': json.dumps(analysis_data.get('profiles', [])),
                'product_insights': json.dumps(analysis_data.get('product_insights', {})),
                'user_behavior': json.dumps(analysis_data.get('user_behavior', {})),
                'recommendations': json.dumps(analysis_data.get('recommendations', []))
            }

            # Add new analysis to list
            existing_analyses.append(new_analysis)

            # Save updated analyses
            fieldnames = ['id', 'user_id', 'topic', 'type', 'title', 'timestamp',
                          'total_posts', 'total_profiles', 'sentiment_distribution',
                          'profiles', 'product_insights', 'user_behavior', 'recommendations']

            with open(analyses_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_analyses)

            self.logger.info(f"Successfully saved analysis for topic: {analysis_data.get('topic')}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")
            return False

    def get_user_analyses(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all analyses for a user"""
        try:
            if not os.path.exists('analyses.csv'):
                return []

            analyses = []
            with open('analyses.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['user_id']) == user_id:
                        try:
                            # Convert JSON strings back to Python objects with error handling
                            sentiment_distribution = json.loads(row.get('sentiment_distribution', '{}'))
                            profiles = json.loads(row.get('profiles', '[]'))
                            product_insights = json.loads(row.get('product_insights', '{}'))
                            user_behavior = json.loads(row.get('user_behavior', '{}'))
                            recommendations = json.loads(row.get('recommendations', '[]'))

                            analysis = {
                                'id': int(row['id']),
                                'user_id': int(row['user_id']),
                                'topic': row.get('topic', ''),
                                'type': row.get('type', 'topic'),
                                'title': row.get('title', ''),
                                'timestamp': row.get('timestamp', ''),
                                'total_posts': int(row.get('total_posts', 0)),
                                'total_profiles': int(row.get('total_profiles', 0)),
                                'sentiment_distribution': sentiment_distribution,
                                'profiles': profiles,
                                'product_insights': product_insights,
                                'user_behavior': user_behavior,
                                'recommendations': recommendations
                            }
                            analyses.append(analysis)
                        except (json.JSONDecodeError, ValueError) as e:
                            self.logger.error(f"Error parsing analysis data: {str(e)}")
                            continue

            # Sort analyses by timestamp in descending order (newest first)
            analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return analyses

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
            profiles_file = 'user_profiles.csv'
            if not os.path.exists(profiles_file):
                return []

            profiles = []
            with open(profiles_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['user_id'] == str(user_id):
                        try:
                            # Parse JSON fields with proper error handling
                            sentiment_counts = json.loads(row.get('sentiment_counts', '{}'))
                            common_hashtags = json.loads(row.get('common_hashtags', '[]'))
                            sample_texts = json.loads(row.get('sample_texts', '[]'))

                            profile = {
                                'id': row.get('id'),
                                'username': row.get('username', ''),
                                'total_posts': int(row.get('total_posts', 0)),
                                'sentiment_counts': sentiment_counts,
                                'common_hashtags': common_hashtags,
                                'sample_texts': sample_texts,
                                'average_confidence': float(row.get('average_confidence', 0.0)),
                                'created_at': row.get('created_at')
                            }
                            profiles.append(profile)
                        except (json.JSONDecodeError, ValueError) as e:
                            self.logger.error(f"Error parsing profile data: {str(e)}")
                            continue

            return profiles

        except Exception as e:
            self.logger.error(f"Error getting user profiles: {str(e)}")
            return []

    def get_quick_stats(self, user_id):
        """Get quick statistics for the dashboard."""
        analyses = self.get_user_analyses(user_id)
        profiles = self.get_user_profiles(user_id)

        return {
            'total_analyses': len(analyses),
            'total_profiles': len(profiles)
        }

    def get_posts_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get posts related to a specific topic"""
        try:
            if self.df is None:
                self.load_dataset()

            # Convert topic to lowercase for case-insensitive matching
            topic = topic.lower()

            # Filter posts that contain the topic in their text
            mask = self.df['text'].str.lower().str.contains(topic, na=False)
            topic_posts = self.df[mask].copy()

            if topic_posts.empty:
                return []

            # Convert DataFrame to list of dictionaries
            posts = []
            for _, row in topic_posts.iterrows():
                post = {
                    'text': row['text'],
                    'username': row['username'],
                    'date': row['date'],
                    'likes': row['likes'],
                    'comments': row['comments']
                }
                posts.append(post)

            return posts

        except Exception as e:
            self.logger.error(f"Error getting posts by topic: {str(e)}")
            return []