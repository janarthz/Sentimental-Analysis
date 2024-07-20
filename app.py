import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def find_text_column(df):
    possible_columns = ['label', 'text']
    for col in possible_columns:
        if col in df.columns:
            return col
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        text_column = find_text_column(df)
        if not text_column:
            return "Error: CSV file does not contain a recognized text column."

        df['Sentiment'] = df[text_column].apply(analyze_sentiment)
        positive_count = (df['Sentiment'] == 'Positive').sum()
        negative_count = (df['Sentiment'] == 'Negative').sum()
        neutral_count = (df['Sentiment'] == 'Neutral').sum()

        overall_sentiment = 'Neutral'
        if positive_count > negative_count:
            overall_sentiment = 'Positive'
        elif negative_count > positive_count:
            overall_sentiment = 'Negative'

        suggestion = ''
        if overall_sentiment == 'Positive':
            suggestion = 'The comments are generally positive. Keep up the good work!'
        elif overall_sentiment == 'Negative':
            suggestion = 'The comments are generally negative. Consider addressing the issues raised.'
        else:
            suggestion = 'The comments are mixed. Try to understand the neutral points and improve accordingly.'

        return render_template('result.html', 
                               positive_count=positive_count, 
                               negative_count=negative_count, 
                               neutral_count=neutral_count,
                               overall_sentiment=overall_sentiment, 
                               suggestion=suggestion)

    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
