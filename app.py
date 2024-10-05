import warnings
import re
import string
import pandas as pd
import joblib
from flask import Flask, render_template, request

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
try:
    Model = joblib.load(r'C:\Users\manas\Spam_news_org(1)\spam_news_detect\fake_model\model.pkl')
except FileNotFoundError:
    print("Model file not found. Please check the file path.")
    Model = None

# Define the home page route
@app.route('/')
def index():
    return render_template("index.html")

# Function to preprocess the text input
def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # Remove special chars
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Define the prediction route
@app.route('/', methods=['POST'])
def pre():
    if request.method == 'POST':
        txt = request.form['txt']  # Get input from form
        txt = wordpre(txt)  # Preprocess the input text
        txt = pd.Series(txt)  # Convert to a pandas Series
        if Model:  # Ensure the model is loaded
            result = Model.predict(txt)  # Make a prediction
            return render_template("index.html", result=result[0])
        else:
            return render_template("index.html", result="Model not found.")
    return '' 

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
