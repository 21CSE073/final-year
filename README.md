# Social Media Analytics Platform

A Flask-based web application for analyzing sentiment and user profiles from social media data. The platform uses machine learning models to provide insights into text sentiment and user behavior patterns.

## Features

- **Sentiment Analysis**: Analyze the sentiment of text using a combination of Naive Bayes and BERT models
- **User Profiling**: Generate detailed user profiles based on topics and hashtags
- **User Authentication**: Secure login and registration system
- **Dashboard**: View and manage your saved analyses
- **Modern UI**: Responsive design with interactive elements

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd social-media-analytics
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. Set up the database:
```python
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

## Configuration

1. Create a `.env` file in the project root:
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
```

2. Place your sentiment dataset in the project root as `sentimentdataset.csv`

## Usage

1. Start the Flask development server:
```bash
flask run
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Create an account or log in to start using the platform

## Project Structure

```
social-media-analytics/
├── app.py              # Main application file
├── models.py           # Database models
├── requirements.txt    # Python dependencies
├── sentimentdataset.csv # Training data
├── templates/         # HTML templates
│   ├── index.html     # Main page
│   ├── login.html     # Login page
│   ├── signup.html    # Registration page
│   └── dashboard.html # User dashboard
└── static/           # Static files (CSS, JS, images)
```

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-Login
- **Frontend**: Bootstrap 5, Font Awesome
- **Machine Learning**: scikit-learn, transformers, torch
- **Data Processing**: pandas, numpy
- **Natural Language Processing**: NLTK

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The BERT model is provided by Hugging Face
- Bootstrap and Font Awesome for the UI components
- The sentiment dataset contributors 